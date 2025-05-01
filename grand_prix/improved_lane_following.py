import math
import numpy as np
import cv2 as cv
from enum import Enum, IntEnum
import sys

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()

# Define color ranges for lane detection
PURPLE = ((110, 100, 100), (140, 255, 255))
ORANGE = ((10, 100, 100), (25, 255, 255))
WHITE = ((0, 0, 200), (179, 20, 255))  # Fixed: upper hue value must be 0-179

# Define windows for LIDAR detection
LEFT_WINDOW = (-135, -45)
RIGHT_WINDOW = (45, 135)
FRONT_WINDOW = (-10, 10)

class LaneFollowingState(Enum):
    NORMAL = 0        # Following two lines normally
    SINGLE_LINE = 1   # Following a single line
    RECOVERY = 2      # In recovery mode, searching for lines
    OBSTACLE_AVOID = 3 # Avoiding an obstacle

class ImprovedLaneFollower:
    def __init__(self):
        # PID controller parameters
        self.kp = 0.6        # Proportional gain (increased for more responsive steering)
        self.ki = 0.1        # Integral gain
        self.kd = 0.3        # Derivative gain (increased for better damping)
        
        # PID controller state
        self.accumulated_error = 0
        self.last_error = 0
        
        # Line following state
        self.state = LaneFollowingState.NORMAL
        self.recovery_counter = 0
        self.last_good_angle = 0
        self.previous_centers = []
        
        # Camera and speed settings
        self.camera_width = rc.camera.get_width()
        self.camera_height = rc.camera.get_height()
        self.max_speed = 0.6  # Increased default max speed
        self.min_contour_area = 30  # Minimum area to consider a contour valid
        
        # Color priority (which color to look for first)
        self.color_priority = [PURPLE, ORANGE, WHITE]  # Added white as a fallback
        
        # Speed and angle
        self.speed = 0
        self.angle = 0
        
        # Line memory
        self.left_line_history = []
        self.right_line_history = []
        self.center_line_history = []
        self.history_max_length = 5

    def start(self):
        """Initialize the lane follower."""
        # Reset all state variables
        self.accumulated_error = 0
        self.last_error = 0
        self.state = LaneFollowingState.NORMAL
        self.recovery_counter = 0
        self.last_good_angle = 0
        self.previous_centers = []
        self.left_line_history = []
        self.right_line_history = []
        self.center_line_history = []
        
        # Set initial speed and angle
        self.speed = 0
        self.angle = 0
        
        # Initialize the car
        rc.drive.set_speed_angle(0, 0)
        rc.drive.set_max_speed(self.max_speed)
        
        print("Improved Lane Following Module initialized")
        rc.set_update_slow_time(0.5)
    
    def pid_control(self, target, current, delta_time):
        """Enhanced PID controller with anti-windup and better output limiting."""
        # Calculate error
        error = target - current
        
        # Update integral term with anti-windup
        self.accumulated_error += error * delta_time
        self.accumulated_error = rc_utils.clamp(self.accumulated_error, -5, 5)  # Prevent excessive windup
        
        # Calculate derivative term with filtering
        if delta_time > 0:
            delta_error = (error - self.last_error) / delta_time
            # Simple low-pass filter to reduce noise in derivative term
            delta_error = 0.7 * delta_error + 0.3 * self.last_error
        else:
            delta_error = 0
        
        # Calculate PID terms
        p_term = self.kp * error
        i_term = self.ki * self.accumulated_error
        d_term = self.kd * delta_error
        
        # Save error for next iteration
        self.last_error = error
        
        # Calculate output with improved limiting
        output = p_term + i_term + d_term
        return rc_utils.clamp(output, -1.0, 1.0)
    
    def preprocess_image(self, image):
        """Preprocess the image to improve line detection."""
        if image is None:
            return None, None
        
        # Crop the image to focus on the road ahead
        crop_floor = ((380, 30), (self.camera_height, self.camera_width))
        cropped_image = rc_utils.crop(image, crop_floor[0], crop_floor[1])
        
        # Convert to HSV for better color filtering
        hsv_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)
        
        # Apply blur to reduce noise
        blurred_image = cv.GaussianBlur(hsv_image, (5, 5), 0)
        
        # Apply contrast enhancement to HSV value channel
        v_channel = blurred_image[:,:,2]
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_v = clahe.apply(v_channel)
        blurred_image[:,:,2] = enhanced_v
        
        return blurred_image, cropped_image
    
    def find_lane_contours(self, hsv_image, original_image):
        """Find lane line contours with improved detection and filtering."""
        left_contour = None
        right_contour = None
        largest_contour = None
        left_center = (0, 0)
        right_center = (0, 0)
        center_point = None
        
        # Try each color in priority order
        for color_range in self.color_priority:
            # Find contours for this color
            contours = rc_utils.find_contours(original_image, color_range[0], color_range[1])
            
            # Skip if no contours found
            if not contours or len(contours) == 0:
                continue
            
            # Filter contours by size
            valid_contours = []
            for contour in contours:
                if contour is not None and cv.contourArea(contour) > self.min_contour_area:
                    valid_contours.append(contour)
            
            if len(valid_contours) == 0:
                continue
            
            # Sort by area (largest first)
            sorted_contours = sorted(valid_contours, key=cv.contourArea, reverse=True)
            
            # Get the largest contour
            largest_contour = sorted_contours[0]
            largest_center = rc_utils.get_contour_center(largest_contour)
            
            # If we have at least two contours, try to identify left and right lines
            if len(sorted_contours) > 1:
                # Sort by x-coordinate to identify left and right lines
                contour_centers = [rc_utils.get_contour_center(c) for c in sorted_contours[:min(5, len(sorted_contours))]]
                sorted_by_x = sorted(zip(contour_centers, sorted_contours[:len(contour_centers)]), key=lambda pair: pair[0][1])
                
                # Left line is the leftmost contour
                left_center, left_contour = sorted_by_x[0]
                # Right line is the rightmost contour
                right_center, right_contour = sorted_by_x[-1]
                
                # Calculate center point between left and right
                if left_center[1] != right_center[1]:  # Make sure they're not the same
                    center_point = (left_center[0], (left_center[1] + right_center[1]) / 2)
                    break
            else:
                # Only one contour found
                left_contour = largest_contour
                left_center = largest_center
                break
        
        return left_contour, right_contour, largest_contour, left_center, right_center, center_point
    
    def update_line_history(self, left_center, right_center, center_point):
        """Update the line history for better tracking."""
        # Update left line history if we have a valid left center
        if left_center[0] != 0 and left_center[1] != 0:
            self.left_line_history.append(left_center)
            if len(self.left_line_history) > self.history_max_length:
                self.left_line_history.pop(0)
        
        # Update right line history if we have a valid right center
        if right_center[0] != 0 and right_center[1] != 0:
            self.right_line_history.append(right_center)
            if len(self.right_line_history) > self.history_max_length:
                self.right_line_history.pop(0)
        
        # Update center point history
        if center_point is not None:
            self.center_line_history.append(center_point)
            if len(self.center_line_history) > self.history_max_length:
                self.center_line_history.pop(0)
    
    def get_smoothed_center(self):
        """Get smoothed center point from history."""
        if len(self.center_line_history) == 0:
            return None
        
        # Calculate average center point
        sum_x = sum(center[0] for center in self.center_line_history)
        sum_y = sum(center[1] for center in self.center_line_history)
        avg_x = sum_x / len(self.center_line_history)
        avg_y = sum_y / len(self.center_line_history)
        
        return (avg_x, avg_y)
    
    def enter_recovery_mode(self, delta_time):
        """Handle recovery when no lines are detected."""
        if self.state != LaneFollowingState.RECOVERY:
            self.state = LaneFollowingState.RECOVERY
            self.recovery_counter = 0
            print("Entering RECOVERY MODE - no lines detected")
        
        # Update recovery counter
        self.recovery_counter += delta_time
        
        # Use last good angle with a sinusoidal search pattern
        search_amplitude = 0.4  # Increased for wider search
        search_frequency = 1.2  # Slightly slower search for better coverage
        search_offset = math.sin(self.recovery_counter * search_frequency * 2 * math.pi) * search_amplitude
        
        # Apply search pattern
        self.angle = self.last_good_angle + search_offset
        
        # Slow down during recovery
        self.speed = 0.7
        
        # Print status occasionally
        if int(self.recovery_counter * 10) % 5 == 0:
            print(f"RECOVERY MODE: Searching with angle {self.angle:.2f}")
        
        # Reset recovery counter after a while to avoid getting stuck in a pattern
        if self.recovery_counter > 3.0:
            self.recovery_counter = 0
    
    def safe_draw_circle(self, image, center, color):
        """Safely draw a circle making sure coordinates are within image bounds."""
        height, width = image.shape[:2]
        
        # rc_utils.draw_circle expects center in (row, col) format
        row = int(center[0])
        col = int(center[1])
        
        # Check if coordinates are within image bounds
        if 0 <= row < height and 0 <= col < width:
            rc_utils.draw_circle(image, (row, col), color)
    
    def follow_lanes(self):
        """Main lane following logic with improved handling."""
        # Get camera image
        image = rc.camera.get_color_image()
        if image is None:
            print("No camera image available!")
            self.speed = 0
            self.angle = 0
            return self.speed, self.angle
        
        # Get delta time for PID control and timing
        delta_time = rc.get_delta_time()
        
        # Preprocess image
        hsv_image, cropped_image = self.preprocess_image(image)
        if cropped_image is None:
            print("Image preprocessing failed!")
            self.speed = 0
            self.angle = 0
            return self.speed, self.angle
        
        # Find lane contours
        left_contour, right_contour, largest_contour, left_center, right_center, center_point = self.find_lane_contours(hsv_image, cropped_image)
        
        # Draw contours for visualization
        display_image = np.copy(cropped_image)
        
        # Get dimensions of the cropped image
        height, width = display_image.shape[:2]
        print(f"Cropped image dimensions: {width}x{height}")
        
        # Check if we have valid contours
        has_left = left_center[0] != 0 and left_center[1] != 0
        has_right = right_center[0] != 0 and right_center[1] != 0
        
        # Update line history
        self.update_line_history(left_center, right_center, center_point)
        
        # Get LIDAR scan for obstacle detection
        scan = rc.lidar.get_samples()
        front_angle, front_distance = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
        left_angle, left_distance = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
        right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
        
        # Default values if no point detected
        front_distance = 1000 if front_distance is None else front_distance
        left_distance = 1000 if left_distance is None else left_distance
        right_distance = 1000 if right_distance is None else right_distance
        
        # Check for obstacles
        obstacle_detected = front_distance < 100 or left_distance < 60 or right_distance < 60
        
        # Main lane following logic
        if has_left and has_right:
            # We have both lines - compute center and follow it
            self.state = LaneFollowingState.NORMAL
            
            # Compute lane width
            lane_width = abs(right_center[1] - left_center[1])
            print(f"Lane width: {lane_width} pixels")
            
            # Use center point between lanes
            target_point = center_point[1]
            
            # Normalize target to range [-1, 1]
            normalized_target = rc_utils.remap_range(target_point, 0, self.camera_width, -1, 1)
            
            # Apply PID control to center the car
            self.angle = self.pid_control(0, normalized_target, delta_time)
            
            # Save last good angle
            basic_angle = normalized_target
            self.last_good_angle = basic_angle
            
            # Set default speed
            self.speed = 1.0
            
            # Reset recovery mode
            self.recovery_counter = 0
            
            # Draw visualization - using direct OpenCV drawing to avoid coordinate issues
            if left_contour is not None:
                rc_utils.draw_contour(display_image, left_contour, (0, 255, 0))
                # Draw circle directly with OpenCV
                cv.circle(display_image, (int(left_center[1]), int(left_center[0])), 5, (0, 255, 0), -1)
            
            if right_contour is not None:
                rc_utils.draw_contour(display_image, right_contour, (0, 0, 255))
                # Draw circle directly with OpenCV
                cv.circle(display_image, (int(right_center[1]), int(right_center[0])), 5, (0, 0, 255), -1)
            
            if center_point is not None:
                # Draw circle directly with OpenCV
                cv.circle(display_image, (int(center_point[1]), int(center_point[0])), 5, (255, 0, 255), -1)
            
        elif has_left or has_right or largest_contour is not None:
            # We have at least one line - follow it
            self.state = LaneFollowingState.SINGLE_LINE
            
            # Use the detected line
            if has_left:
                center_point = left_center
                rc_utils.draw_contour(display_image, left_contour, (0, 255, 0))
                # Draw circle directly with OpenCV
                cv.circle(display_image, (int(left_center[1]), int(left_center[0])), 5, (0, 255, 0), -1)
                # Since we only see left line, we should be on the right side of it
                target_point = center_point[1] + 100  # Aim to the right of the line
            elif has_right:
                center_point = right_center
                rc_utils.draw_contour(display_image, right_contour, (0, 0, 255))
                # Draw circle directly with OpenCV
                cv.circle(display_image, (int(right_center[1]), int(right_center[0])), 5, (0, 0, 255), -1)
                # Since we only see right line, we should be on the left side of it
                target_point = center_point[1] - 100  # Aim to the left of the line
            else:
                # Use largest contour as fallback
                center_point = rc_utils.get_contour_center(largest_contour)
                rc_utils.draw_contour(display_image, largest_contour, (255, 255, 0))
                # Draw circle directly with OpenCV
                cv.circle(display_image, (int(center_point[1]), int(center_point[0])), 5, (255, 255, 0), -1)
                # Target point is just the center
                target_point = center_point[1]
            
            print(f"Following single line at x={target_point}")
            
            # Normalize target to range [-1, 1]
            normalized_target = rc_utils.remap_range(target_point, 0, self.camera_width, -1, 1)
            
            # Apply PID control
            self.angle = self.pid_control(0, normalized_target, delta_time)
            
            # Save last good angle
            self.last_good_angle = normalized_target
            
            # Set speed
            self.speed = 0.9  # Slightly lower speed when following single line
            
            # Reset recovery mode
            self.recovery_counter = 0
            
        else:
            # No lines detected - enter recovery mode
            self.enter_recovery_mode(delta_time)
        
        # Enhanced wall avoidance logic from grand_prix_Finished.py
        if obstacle_detected:
            self.state = LaneFollowingState.OBSTACLE_AVOID
            
            # Initialize wall avoidance angle
            wall_avoidance_angle = 0
            
            # If wall is too close on the left, turn right
            if left_distance < 60:
                wall_avoidance_angle = rc_utils.remap_range(left_distance, 20, 60, 0.7, 0.2)
                print(f"Wall on left: {left_distance:.1f}cm, adding angle: {wall_avoidance_angle:.2f}")
            
            # If wall is too close on the right, turn left
            elif right_distance < 60:
                wall_avoidance_angle = rc_utils.remap_range(right_distance, 20, 60, -0.7, -0.2)
                print(f"Wall on right: {right_distance:.1f}cm, adding angle: {wall_avoidance_angle:.2f}")
            
            # If wall is directly in front, make a stronger turn
            if front_distance < 100:
                # Turn away from the closest side wall, or choose left by default if equal
                if left_distance < right_distance:
                    wall_avoidance_angle = 1  # Turn right
                else:
                    wall_avoidance_angle = -1  # Turn left
                
                # Slow down when approaching a wall
                self.speed = rc_utils.remap_range(front_distance, 30, 100, 0.5, 1.0)
                print(f"Wall ahead: {front_distance:.1f}cm, strong avoid: {wall_avoidance_angle:.2f}")
            
            # Combine lane following angle with wall avoidance
            # Lane following gets priority, but wall avoidance can override if needed
            if abs(wall_avoidance_angle) > 0.1:
                # Blend the angles, with more weight to wall avoidance when walls are very close
                blend_factor = rc_utils.remap_range(min(left_distance, right_distance, front_distance), 
                                                20, 60, 0.8, 0.3)
                blend_factor = rc_utils.clamp(blend_factor, 0, 0.8)
                
                # Apply blending
                self.angle = self.angle * (1 - blend_factor) + wall_avoidance_angle * blend_factor
                
                # Apply additional steering bias for sharper turns

        
        # Apply speed adjustment based on steering angle
        # Slow down in sharp turns for stability
        speed_factor = 1.0 - abs(self.angle) * 0.5
        self.speed *= max(0.5, speed_factor)
        
        # Clamp controls to valid range
        self.angle = rc_utils.clamp(self.angle, -1.0, 1.0)
        self.speed = rc_utils.clamp(self.speed, -1.0, 1.0)
        
        # Show processed image
        rc.display.show_color_image(display_image)
        
        return self.speed, self.angle
    
    def update(self):
        """Update function to be called each frame."""
        # Update lane following
        speed, angle = self.follow_lanes()
        speed = rc_utils.clamp(speed, -1.0, 1.0)
        angle = rc_utils.clamp(angle, -1.0, 1.0)
        # Apply final controls
        rc.drive.set_speed_angle(speed, angle)
    
    def update_slow(self):
        """Update function to be called at a slower rate."""
        # Print state info
        states = {
            LaneFollowingState.NORMAL: "NORMAL - Following both lines",
            LaneFollowingState.SINGLE_LINE: "SINGLE_LINE - Following one line",
            LaneFollowingState.RECOVERY: f"RECOVERY - Searching for {self.recovery_counter:.1f}s",
            LaneFollowingState.OBSTACLE_AVOID: "OBSTACLE_AVOID - Avoiding obstacle"
        }
        
        print(f"Lane Following State: {states[self.state]}")
        print(f"Speed: {self.speed:.2f}, Angle: {self.angle:.2f}")
        print(f"Camera crop: Starting at 320px from top")

# Global instance that can be imported
lane_follower = ImprovedLaneFollower()

# Function to start the lane follower
def start():
    """Initialize and start the lane follower."""
    lane_follower.start()

# Function to update the lane follower (to be called each frame)
def update():
    """Update the lane follower."""
    lane_follower.update()

# Function to update the lane follower at a slower rate
def update_slow():
    """Update the lane follower at a slower rate."""
    lane_follower.update_slow()

# Main function for standalone testing
if __name__ == "__main__":
    # Set update and update_slow handlers
    rc.set_start_update(start, update, update_slow)
    
    # Set update_slow time
    rc.set_update_slow_time(0.5)
    
    # Start the main loop
    rc.go() 