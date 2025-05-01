import math
import numpy as np
import cv2 as cv
from enum import Enum
import sys
from collections import deque
import time
import traceback

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

# Create the racecar object
rc = racecar_core.create_racecar()

# Constants for lane detection - using HSV color ranges from grand_prix_Finished.py
PURPLE = ((125, 100, 100), (145, 255, 255))  # Updated from grand_prix_Finished.py
ORANGE = ((5, 100, 100), (25, 255, 255))     # Updated from grand_prix_Finished.py
WHITE = ((0, 0, 200), (179, 20, 255))        # Keeping this from improved_lane_following.py for white detection

# Additional colors from grand_prix_Finished.py if needed
RED = ((170, 50, 50), (10, 255, 255))
BLUE = ((100, 150, 50), (110, 255, 255))
GREEN = ((60, 200, 200), (80, 255, 255))
YELLOW = ((20, 100, 100), (40, 255, 255))

# LIDAR windows
LEFT_WINDOW = (-135, -45)
RIGHT_WINDOW = (45, 135)
FRONT_WINDOW = (-15, 15)

# Image processing constants
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 5

# Enhanced Speed Control Parameters
MAX_SPEED = 1.0      # Maximum possible speed (increased from 0.8)
MIN_SPEED = 0.3      # Minimum speed for normal operation
CRUISE_SPEED = 0.7   # Default cruising speed
TURN_SPEED = 0.5     # Speed during sharp turns
RECOVERY_SPEED = 0.4 # Speed during recovery mode
WALL_FOLLOW_SPEED = 0.5  # Speed during wall following
EMERGENCY_SPEED = 0.2    # Speed during emergency situations
ACCELERATION_RATE = 0.2  # How quickly to increase speed (units/sec)
DECELERATION_RATE = 0.4  # How quickly to decrease speed (units/sec)
CURVE_SPEED_FACTOR = 0.8 # How much to reduce speed in curves
LANE_WIDTH_FACTOR = 0.8  # How much lane width affects speed

# Other control parameters (unchanged)
TURN_FACTOR = 0.7
OBSTACLE_DISTANCE_THRESHOLD = 100  # cm
WALL_DISTANCE_THRESHOLD = 60       # cm
EMERGENCY_THRESHOLD = 30           # cm

class LaneState(Enum):
    """State of the lane follower."""
    NORMAL = 0          # Both lanes detected
    SINGLE_LEFT = 1     # Only left lane detected
    SINGLE_RIGHT = 2    # Only right lane detected
    RECOVERY = 3        # No lanes detected
    WALL_FOLLOW = 4     # Using walls for guidance
    EMERGENCY = 5       # Emergency obstacle avoidance

class AdvancedLaneFollower:
    def __init__(self):
        # PID controller state
        self.kp = 0.7  # Increased proportional gain for more responsive steering
        self.ki = 0.1  # Small integral gain to handle steady-state errors
        self.kd = 0.3  # Derivative gain for dampening
        self.error_sum = 0.0
        self.last_error = 0.0
        
        # Kalman filter parameters
        self.kalman_state = np.zeros(4)  # [left_x, left_vx, right_x, right_vx]
        self.kalman_covariance = np.eye(4) * 100  # Initial uncertainty
        self.process_noise = 1.0  # Process noise
        self.measurement_noise = 10.0  # Measurement noise
        
        # Lane tracking history
        self.lane_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10),
            'center': deque(maxlen=10)
        }
        
        # State management
        self.state = LaneState.NORMAL
        self.recovery_counter = 0
        self.last_detection_time = time.time()
        self.recovery_pattern = 0
        
        # Camera dimensions
        self.camera_width = rc.camera.get_width()
        self.camera_height = rc.camera.get_height()
        
        # Control variables
        self.speed = 0.0
        self.angle = 0.0
        self.target_speed = CRUISE_SPEED  # Target speed (desired speed)
        self.current_speed = 0.0  # Current actual speed
        self.max_speed = MAX_SPEED
        
        # Enhanced speed control
        self.acceleration = ACCELERATION_RATE
        self.deceleration = DECELERATION_RATE
        self.curve_speed_reduction = CURVE_SPEED_FACTOR
        self.last_update_time = time.time()
        self.speed_history = deque(maxlen=5)  # For speed smoothing
        self.curve_history = deque(maxlen=10)  # For curve detection
        
        # Region of interest parameters - using crop region from grand_prix_Finished.py
        self.roi_top = 360  # Matches CROP_FLOOR in grand_prix_Finished.py
        self.roi_bottom = None  # Will be set to image height
        self.roi_width = None  # Will be set based on image width
        
        # Wall following parameters
        self.target_wall_distance = 50  # Target distance from wall in cm
        self.current_wall_side = None  # 'left' or 'right'
        
        # Color priority list (order of colors to look for)
        self.color_priority = [PURPLE, ORANGE, WHITE]
        
        # Path prediction for look-ahead speed control
        self.predicted_path = []  # Store predicted path points
        self.path_difficulty = 0.0  # Estimate of path difficulty (0.0-1.0)
    
    def start(self):
        """Initialize the lane follower."""
        print("Advanced Lane Follower starting...")
        
        # Reset state
        self.state = LaneState.NORMAL
        self.error_sum = 0.0
        self.last_error = 0.0
        self.recovery_counter = 0
        
        # Reset control
        self.speed = 0.0
        self.current_speed = 0.0
        self.target_speed = CRUISE_SPEED
        self.angle = 0.0
        
        # Reset history
        self.lane_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10),
            'center': deque(maxlen=10)
        }
        self.speed_history.clear()
        self.curve_history.clear()
        
        # Initialize camera dimensions
        self.camera_width = rc.camera.get_width()
        self.camera_height = rc.camera.get_height()
        self.roi_bottom = self.camera_height
        self.roi_width = self.camera_width
        
        # Configure the car
        rc.drive.set_speed_angle(0, 0)
        rc.drive.set_max_speed(self.max_speed)
        
        print(f"Camera dimensions: {self.camera_width}x{self.camera_height}")
        print("Advanced Lane Follower initialized")
    
    def update_speed_control(self, dt):
        """
        Apply sophisticated speed control based on current state and conditions.
        """
        # Determine target speed based on current state
        if self.state == LaneState.EMERGENCY:
            # Emergency - very slow
            new_target_speed = EMERGENCY_SPEED
        elif self.state == LaneState.RECOVERY:
            # Recovery mode - reduced speed
            new_target_speed = RECOVERY_SPEED
        elif self.state == LaneState.WALL_FOLLOW:
            # Wall following - moderate speed
            new_target_speed = WALL_FOLLOW_SPEED
        elif self.state == LaneState.NORMAL:
            # Normal lane following - use curve-based speed
            # Calculate path curvature from recent steering angles
            if len(self.curve_history) >= 3:
                # Calculate mean absolute steering angle as a measure of curvature
                mean_abs_angle = sum(abs(angle) for angle in self.curve_history) / len(self.curve_history)
                # More steering = More curvature = Lower speed
                curvature_factor = 1.0 - mean_abs_angle * self.curve_speed_reduction
                curvature_factor = rc_utils.clamp(curvature_factor, 0.5, 1.0)
                
                # Adjust for lane width if available (wider = safer = faster)
                lane_width_factor = 1.0
                if len(self.lane_history['left']) > 0 and len(self.lane_history['right']) > 0:
                    left_x = self.lane_history['left'][-1][1]
                    right_x = self.lane_history['right'][-1][1]
                    lane_width = abs(right_x - left_x)
                    
                    # Normalize lane width to a 0-1 factor (80-300 pixels → 0.5-1.0)
                    lane_width_factor = rc_utils.remap_range(lane_width, 80, 300, 0.5, 1.0)
                    lane_width_factor = rc_utils.clamp(lane_width_factor, 0.5, 1.0)
                
                # Combine factors
                new_target_speed = CRUISE_SPEED * curvature_factor * lane_width_factor
            else:
                # Not enough history, use safe default
                new_target_speed = CRUISE_SPEED * 0.8
        else:
            # Single lane detection - moderately reduced speed
            new_target_speed = CRUISE_SPEED * 0.7
        
        # Look-ahead speed adjustment based on predicted path difficulty
        if hasattr(self, 'path_difficulty') and self.path_difficulty > 0:
            # Reduce speed further if upcoming path looks difficult
            look_ahead_factor = 1.0 - self.path_difficulty * 0.5
            new_target_speed *= look_ahead_factor
        
        # Apply speed limits
        new_target_speed = rc_utils.clamp(new_target_speed, MIN_SPEED, MAX_SPEED)
        self.target_speed = new_target_speed
        
        # Gradually adjust current speed towards target speed using acceleration/deceleration
        if self.current_speed < self.target_speed:
            # Accelerate
            self.current_speed += self.acceleration * dt
            if self.current_speed > self.target_speed:
                self.current_speed = self.target_speed
        elif self.current_speed > self.target_speed:
            # Decelerate
            self.current_speed -= self.deceleration * dt
            if self.current_speed < self.target_speed:
                self.current_speed = self.target_speed
        
        # Apply smoothing to speed changes
        self.speed_history.append(self.current_speed)
        if len(self.speed_history) > 0:
            self.speed = sum(self.speed_history) / len(self.speed_history)
        else:
            self.speed = self.current_speed
            
        # Store current angle in curve history
        self.curve_history.append(self.angle)
        
        # Debug info
        print(f"Speed Control: target={self.target_speed:.2f}, current={self.current_speed:.2f}, applied={self.speed:.2f}")
    
    def predict_path_difficulty(self, left_center, right_center):
        """
        Analyze upcoming path and predict its difficulty.
        Returns a difficulty score from 0.0 (easy) to 1.0 (difficult).
        """
        difficulty = 0.0
        
        # 1. Check lane width stability
        if len(self.lane_history['left']) > 5 and len(self.lane_history['right']) > 5:
            # Calculate variance in lane width
            widths = []
            for i in range(min(len(self.lane_history['left']), len(self.lane_history['right']))):
                left_x = self.lane_history['left'][i][1]
                right_x = self.lane_history['right'][i][1]
                widths.append(abs(right_x - left_x))
            
            if len(widths) > 0:
                mean_width = sum(widths) / len(widths)
                if mean_width > 0:
                    width_variance = sum((w - mean_width)**2 for w in widths) / len(widths)
                    normalized_variance = min(width_variance / (mean_width * 0.5), 1.0)
                    difficulty += normalized_variance * 0.3  # Weight this factor by 0.3
        
        # 2. Check steering angle stability
        if len(self.curve_history) > 5:
            angles = list(self.curve_history)
            mean_angle = sum(angles) / len(angles)
            angle_variance = sum((a - mean_angle)**2 for a in angles) / len(angles)
            normalized_angle_variance = min(angle_variance / 0.3, 1.0)
            difficulty += normalized_angle_variance * 0.4  # Weight this factor by 0.4
        
        # 3. Check for rapid lane shifts
        if len(self.lane_history['center']) > 3:
            centers = [c[1] for c in self.lane_history['center']]
            center_shifts = [abs(centers[i] - centers[i-1]) for i in range(1, len(centers))]
            if center_shifts:
                max_shift = max(center_shifts)
                normalized_shift = min(max_shift / 50, 1.0)  # Normalize to 0-1
                difficulty += normalized_shift * 0.3  # Weight this factor by 0.3
        
        self.path_difficulty = difficulty
        return difficulty
    
    def enhanced_image_processing(self, image):
        """
        Multi-stage image processing pipeline for robust lane detection.
        Returns processed image and extracted lane lines.
        """
        if image is None:
            return None, None, None
        
        # Crop to region of interest (lower portion of image)
        top_left = (self.roi_top, 0)
        bottom_right = (self.roi_bottom, self.roi_width)
        cropped = rc_utils.crop(image, top_left, bottom_right)
        
        # Create a copy for visualization
        display_image = np.copy(cropped)
        
        # Stage 1: Process in HSV color space for lane colors
        hsv_image = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
        
        # Apply contrast enhancement to V channel
        v_channel = hsv_image[:,:,2]
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_v = clahe.apply(v_channel)
        hsv_image[:,:,2] = enhanced_v
        
        # Stage 2: Multi-color detection
        # Create binary masks for each color in our priority list
        combined_mask = None
        
        for color_range in self.color_priority:
            color_mask = cv.inRange(hsv_image, color_range[0], color_range[1])
            
            if combined_mask is None:
                combined_mask = color_mask
            else:
                combined_mask = cv.bitwise_or(combined_mask, color_mask)
        
        # Stage 3: Edge detection
        # Convert to grayscale for edge detection
        gray_image = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray_image, (5, 5), 0)
        
        # Adaptive thresholding for varying light conditions
        # This helps detect lanes even with shadows or bright areas
        binary_adaptive = cv.adaptiveThreshold(
            blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
        )
        
        # Combine color and adaptive threshold results
        lane_mask = cv.bitwise_or(combined_mask, binary_adaptive)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv.morphologyEx(lane_mask, cv.MORPH_CLOSE, kernel)
        lane_mask = cv.morphologyEx(lane_mask, cv.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv.findContours(lane_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and organize by position (left or right)
        min_contour_area = 100
        valid_contours = [c for c in contours if cv.contourArea(c) > min_contour_area]
        
        if not valid_contours:
            return display_image, None, None
        
        # Sort contours by x-coordinate of centroid
        contour_centers = [rc_utils.get_contour_center(c) for c in valid_contours]
        contour_with_centers = list(zip(valid_contours, contour_centers))
        sorted_contours = sorted(contour_with_centers, key=lambda x: x[1][1])  # Sort by x coordinate
        
        # We'll consider the leftmost and rightmost large contours as lane markers
        left_lane = right_lane = None
        left_center = right_center = None
        
        if len(sorted_contours) >= 2:
            # We have at least two contours - take leftmost and rightmost
            left_lane, left_center = sorted_contours[0]
            right_lane, right_center = sorted_contours[-1]
        elif len(sorted_contours) == 1:
            # We only have one contour - determine if it's left or right based on position
            contour, center = sorted_contours[0]
            image_center_x = display_image.shape[1] // 2
            
            if center[1] < image_center_x:  # Contour is on the left side
                left_lane, left_center = contour, center
            else:  # Contour is on the right side
                right_lane, right_center = contour, center
        
        # Draw the detected lanes on the display image
        if left_lane is not None:
            cv.drawContours(display_image, [left_lane], -1, (0, 255, 0), 2)
            cv.circle(display_image, (int(left_center[1]), int(left_center[0])), 5, (0, 255, 0), -1)
        
        if right_lane is not None:
            cv.drawContours(display_image, [right_lane], -1, (0, 0, 255), 2)
            cv.circle(display_image, (int(right_center[1]), int(right_center[0])), 5, (0, 0, 255), -1)
        
        return display_image, left_center, right_center
    
    def update_lane_history(self, left_center, right_center):
        """
        Update lane position history and apply Kalman filtering.
        """
        current_time = time.time()
        dt = current_time - self.last_detection_time
        self.last_detection_time = current_time
        
        # Update lane history with new detections
        if left_center is not None:
            self.lane_history['left'].append(left_center)
        
        if right_center is not None:
            self.lane_history['right'].append(right_center)
        
        # Calculate center point if both lanes are detected
        if left_center is not None and right_center is not None:
            center_x = (left_center[1] + right_center[1]) / 2
            center_y = (left_center[0] + right_center[0]) / 2
            center_point = (center_y, center_x)
            self.lane_history['center'].append(center_point)
        
        # Apply Kalman filter prediction step
        if dt > 0:
            # State transition matrix
            F = np.array([
                [1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1]
            ])
            
            # Predict next state
            self.kalman_state = F @ self.kalman_state
            
            # Update covariance
            Q = np.eye(4) * self.process_noise * dt
            self.kalman_covariance = F @ self.kalman_covariance @ F.T + Q
        
        # Kalman filter update step if measurements are available
        if left_center is not None or right_center is not None:
            # Measurement vector
            z = np.zeros(2)
            H = np.zeros((2, 4))
            R = np.eye(2) * self.measurement_noise
            
            if left_center is not None and right_center is not None:
                # Both lanes detected
                z = np.array([left_center[1], right_center[1]])
                H = np.array([
                    [1, 0, 0, 0],  # Measure left_x
                    [0, 0, 1, 0]   # Measure right_x
                ])
            elif left_center is not None:
                # Only left lane detected
                z = np.array([left_center[1], self.kalman_state[2]])
                H = np.array([
                    [1, 0, 0, 0],  # Measure left_x
                    [0, 0, 0, 0]   # No measurement for right_x
                ])
            elif right_center is not None:
                # Only right lane detected
                z = np.array([self.kalman_state[0], right_center[1]])
                H = np.array([
                    [0, 0, 0, 0],  # No measurement for left_x
                    [0, 0, 1, 0]   # Measure right_x
                ])
            
            # Kalman gain
            S = H @ self.kalman_covariance @ H.T + R
            K = self.kalman_covariance @ H.T @ np.linalg.inv(S)
            
            # Update state
            y = z - H @ self.kalman_state
            self.kalman_state = self.kalman_state + K @ y
            
            # Update covariance
            I = np.eye(4)
            self.kalman_covariance = (I - K @ H) @ self.kalman_covariance
    
    def get_filtered_lane_positions(self):
        """
        Get filtered lane positions from Kalman filter.
        Returns left_x, right_x, and center_x in image coordinates.
        """
        left_x = self.kalman_state[0]
        right_x = self.kalman_state[2]
        center_x = (left_x + right_x) / 2
        
        # Validate Kalman output against recent history for robustness
        if self.lane_history['left'] and abs(left_x - self.lane_history['left'][-1][1]) > 100:
            # Kalman prediction too far from last detection, use history instead
            left_x = self.lane_history['left'][-1][1]
        
        if self.lane_history['right'] and abs(right_x - self.lane_history['right'][-1][1]) > 100:
            # Kalman prediction too far from last detection, use history instead
            right_x = self.lane_history['right'][-1][1]
        
        return left_x, right_x, center_x
    
    def adaptive_pid_control(self, target, current, dt):
        """
        PID controller with adaptive gains based on speed and confidence.
        """
        # Calculate error
        error = target - current
        
        # Update integral term with anti-windup
        max_integral = 10.0 / self.ki if self.ki > 0 else 10.0
        self.error_sum += error * dt
        self.error_sum = rc_utils.clamp(self.error_sum, -max_integral, max_integral)
        
        # Calculate derivative term with filtering
        d_error = 0
        if dt > 0:
            d_error = (error - self.last_error) / dt
            # Simple low-pass filter
            d_error = 0.8 * d_error + 0.2 * (self.last_error / dt if dt > 0 else 0)
        
        # Calculate PID terms
        p_term = self.kp * error
        i_term = self.ki * self.error_sum
        d_term = self.kd * d_error
        
        # Save error for next iteration
        self.last_error = error
        
        # Calculate output
        output = p_term + i_term + d_term
        return rc_utils.clamp(output, -1.0, 1.0)
    
    def process_lidar_for_wall_following(self):
        """
        Process LIDAR data for wall detection and following.
        """
        # Get LIDAR samples
        scan = rc.lidar.get_samples()
        
        # Find closest points in each window
        front_angle, front_distance = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
        left_angle, left_distance = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
        right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
        
        # Set default values if no points detected
        front_distance = 1000 if front_distance is None else front_distance
        left_distance = 1000 if left_distance is None else left_distance
        right_distance = 1000 if right_distance is None else right_distance
        
        # Determine wall following side if not already set
        if self.current_wall_side is None:
            if left_distance < right_distance and left_distance < 200:
                self.current_wall_side = 'left'
            elif right_distance < 200:
                self.current_wall_side = 'right'
        
        return front_distance, left_distance, right_distance
    
    def calculate_wall_following_angle(self, front_distance, left_distance, right_distance):
        """
        Calculate steering angle for wall following based on LIDAR.
        """
        # Initialize wall following angle
        wall_angle = 0.0
        
        # If no walls detected at a reasonable distance, return 0
        if left_distance > 200 and right_distance > 200 and front_distance > 200:
            return 0.0
        
        # Emergency front collision avoidance
        if front_distance < EMERGENCY_THRESHOLD:
            # Sharp turn away from the closest side
            if left_distance < right_distance:
                return 1.0  # Turn hard right
            else:
                return -1.0  # Turn hard left
        
        # Calculate angle based on current wall side
        if self.current_wall_side == 'left':
            # Follow left wall
            error = left_distance - self.target_wall_distance
            # Positive error means we're too far from wall -> turn left (negative angle)
            # Negative error means we're too close to wall -> turn right (positive angle)
            wall_angle = -rc_utils.clamp(error / 50, -0.8, 0.8)
            
            # Adjust for front obstacles
            if front_distance < WALL_DISTANCE_THRESHOLD:
                # Blend in a right turn
                front_factor = rc_utils.remap_range(front_distance, 
                                                  EMERGENCY_THRESHOLD, 
                                                  WALL_DISTANCE_THRESHOLD, 
                                                  1.0, 0.0)
                wall_angle = rc_utils.clamp(wall_angle + front_factor * 0.8, -1.0, 1.0)
                
        elif self.current_wall_side == 'right':
            # Follow right wall
            error = right_distance - self.target_wall_distance
            # Positive error means we're too far from wall -> turn right (positive angle)
            # Negative error means we're too close to wall -> turn left (negative angle)
            wall_angle = rc_utils.clamp(error / 50, -0.8, 0.8)
            
            # Adjust for front obstacles
            if front_distance < WALL_DISTANCE_THRESHOLD:
                # Blend in a left turn
                front_factor = rc_utils.remap_range(front_distance, 
                                                  EMERGENCY_THRESHOLD, 
                                                  WALL_DISTANCE_THRESHOLD, 
                                                  1.0, 0.0)
                wall_angle = rc_utils.clamp(wall_angle - front_factor * 0.8, -1.0, 1.0)
        else:
            # No wall side selected, basic obstacle avoidance
            if front_distance < OBSTACLE_DISTANCE_THRESHOLD:
                # Turn away from closest side
                if left_distance < right_distance:
                    wall_angle = rc_utils.remap_range(front_distance, 
                                                   EMERGENCY_THRESHOLD, 
                                                   OBSTACLE_DISTANCE_THRESHOLD, 
                                                   1.0, 0.3)  # Turn right
                else:
                    wall_angle = rc_utils.remap_range(front_distance, 
                                                   EMERGENCY_THRESHOLD, 
                                                   OBSTACLE_DISTANCE_THRESHOLD, 
                                                   -1.0, -0.3)  # Turn left
        
        return wall_angle
    
    def recovery_strategy(self, dt):
        """
        Implement tiered recovery strategy when lanes are lost.
        """
        # Update recovery counter
        self.recovery_counter += dt
        
        # Level 1 (0-1s): Maintain last known direction
        if self.recovery_counter < 1.0:
            self.angle = self.last_error  # Use last error as the steering angle
            self.speed = MIN_SPEED
            return
        
        # Level 2 (1-3s): Incremental search
        elif self.recovery_counter < 3.0:
            # Oscillate with increasing amplitude
            amplitude = 0.2 + (self.recovery_counter - 1.0) * 0.2  # 0.2 to 0.6
            frequency = 2.0  # cycles per second
            self.angle = amplitude * math.sin(frequency * self.recovery_counter * 2 * math.pi)
            self.speed = MIN_SPEED
            return
        
        # Level 3 (3-6s): Check LIDAR and use wall following if walls detected
        elif self.recovery_counter < 6.0:
            front_distance, left_distance, right_distance = self.process_lidar_for_wall_following()
            
            # If walls are detected, switch to wall following
            if (left_distance < WALL_DISTANCE_THRESHOLD or 
                right_distance < WALL_DISTANCE_THRESHOLD or
                front_distance < OBSTACLE_DISTANCE_THRESHOLD):
                self.state = LaneState.WALL_FOLLOW
                # Will be handled in next update
                return
            
            # Continue more aggressive search pattern
            amplitude = 0.7
            frequency = 1.0
            self.angle = amplitude * math.sin(frequency * self.recovery_counter * 2 * math.pi)
            self.speed = MIN_SPEED * 0.7  # Even slower
            return
        
        # Level 4 (>6s): Reset recovery and try pattern-based search
        else:
            self.recovery_counter = 0
            
            # Cycle through different search patterns
            self.recovery_pattern = (self.recovery_pattern + 1) % 4
            
            if self.recovery_pattern == 0:
                # Try straight ahead
                self.angle = 0.0
            elif self.recovery_pattern == 1:
                # Try left
                self.angle = -0.5
            elif self.recovery_pattern == 2:
                # Try straight again
                self.angle = 0.0
            else:  # pattern == 3
                # Try right
                self.angle = 0.5
            
            self.speed = MIN_SPEED * 0.5  # Very slow
            return
    
    def follow_lanes(self, image):
        """
        Main lane following logic with integrated wall following and recovery.
        """
        # Get current time for timing information
        current_time = time.time()
        dt = current_time - self.last_detection_time
        
        # Get camera image
        color_image = rc.camera.get_color_image()
        if color_image is None:
            print("No camera image available!")
            self.speed = 0
            self.angle = 0
            return
        
        # Process image to detect lanes
        display_image, left_center, right_center = self.enhanced_image_processing(color_image)
        if display_image is None:
            print("Image processing failed!")
            self.speed = 0
            self.angle = 0
            return
        
        # Get LIDAR data
        front_distance, left_distance, right_distance = self.process_lidar_for_wall_following()
        
        # Check if we have emergency obstacles
        emergency_obstacle = front_distance < EMERGENCY_THRESHOLD
        
        if emergency_obstacle:
            # Emergency obstacle avoidance takes precedence
            self.state = LaneState.EMERGENCY
            if left_distance < right_distance:
                self.angle = 1.0  # Turn hard right
            else:
                self.angle = -1.0  # Turn hard left
            
            self.speed = MIN_SPEED * 0.5  # Very slow
            
            # Show visualization
            cv.putText(display_image, "EMERGENCY AVOID", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rc.display.show_color_image(display_image)
            return
        
        # Update lane tracking state based on what was detected
        if left_center is not None and right_center is not None:
            # Both lanes detected
            self.state = LaneState.NORMAL
            self.recovery_counter = 0
            
            # Update lane history and Kalman filter
            self.update_lane_history(left_center, right_center)
            
            # Calculate center point between lanes
            center_x = (left_center[1] + right_center[1]) / 2
            
            # Calculate target (image center) and error
            image_center_x = display_image.shape[1] / 2
            error = (center_x - image_center_x) / image_center_x  # Normalize to [-1, 1]
            
            # Apply PID control
            self.angle = self.adaptive_pid_control(0, error, dt)
            
            # Set speed based on confidence and steering angle
            lane_width = abs(right_center[1] - left_center[1])
            if lane_width > 50 and lane_width < 400:  # Reasonable lane width
                speed_factor = 1.0 - abs(self.angle) * 0.7  # Reduce speed in turns
                self.speed = MAX_SPEED * speed_factor
            else:
                # Abnormal lane width, go slower
                self.speed = MIN_SPEED
            
            # Show visualization
            cv.putText(display_image, "NORMAL: Both Lanes", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.line(display_image, (int(center_x), display_image.shape[0]), 
                   (int(center_x), 0), (255, 0, 255), 2)
            
        elif left_center is not None:
            # Only left lane detected
            self.state = LaneState.SINGLE_LEFT
            self.recovery_counter = 0
            
            # Update lane history and Kalman filter
            self.update_lane_history(left_center, None)
            
            # Use left lane plus offset as reference
            offset = 150  # Estimated half lane width
            target_x = left_center[1] + offset
            
            # Calculate error from image center
            image_center_x = display_image.shape[1] / 2
            error = (target_x - image_center_x) / image_center_x
            
            # Apply PID control
            self.angle = self.adaptive_pid_control(0, error, dt)
            
            # Go slower when only one lane is visible
            speed_factor = 1.0 - abs(self.angle) * 0.8
            self.speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * 0.7 * speed_factor
            
            # Show visualization
            cv.putText(display_image, "SINGLE: Left Lane", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.line(display_image, (int(target_x), display_image.shape[0]), 
                   (int(target_x), 0), (255, 255, 0), 2)
            
        elif right_center is not None:
            # Only right lane detected
            self.state = LaneState.SINGLE_RIGHT
            self.recovery_counter = 0
            
            # Update lane history and Kalman filter
            self.update_lane_history(None, right_center)
            
            # Use right lane minus offset as reference
            offset = 150  # Estimated half lane width
            target_x = right_center[1] - offset
            
            # Calculate error from image center
            image_center_x = display_image.shape[1] / 2
            error = (target_x - image_center_x) / image_center_x
            
            # Apply PID control
            self.angle = self.adaptive_pid_control(0, error, dt)
            
            # Go slower when only one lane is visible
            speed_factor = 1.0 - abs(self.angle) * 0.8
            self.speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * 0.7 * speed_factor
            
            # Show visualization
            cv.putText(display_image, "SINGLE: Right Lane", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.line(display_image, (int(target_x), display_image.shape[0]), 
                   (int(target_x), 0), (255, 255, 0), 2)
            
        elif (left_distance < WALL_DISTANCE_THRESHOLD or 
              right_distance < WALL_DISTANCE_THRESHOLD or
              front_distance < OBSTACLE_DISTANCE_THRESHOLD):
            # No lanes but walls detected - use wall following
            self.state = LaneState.WALL_FOLLOW
            
            # Calculate wall following angle
            self.angle = self.calculate_wall_following_angle(front_distance, left_distance, right_distance)
            
            # Set slower speed for wall following
            speed_factor = 1.0 - abs(self.angle) * 0.9
            self.speed = MIN_SPEED * speed_factor
            
            # Show visualization
            cv.putText(display_image, f"WALL FOLLOW: {self.current_wall_side}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Add LIDAR info to display
            cv.putText(display_image, f"F:{front_distance:.1f} L:{left_distance:.1f} R:{right_distance:.1f}", 
                      (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
        else:
            # No lanes or walls detected - enter recovery mode
            self.state = LaneState.RECOVERY
            
            # Try to recover using tiered strategy
            self.recovery_strategy(dt)
            
            # Show visualization
            cv.putText(display_image, f"RECOVERY: {self.recovery_counter:.1f}s", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Ensure controls are within valid ranges
        self.angle = rc_utils.clamp(self.angle, -1.0, 1.0)
        self.speed = rc_utils.clamp(self.speed, -1.0, 1.0)
        
        # Show state information on display
        cv.putText(display_image, f"Speed: {self.speed:.2f} Angle: {self.angle:.2f}", 
                  (10, display_image.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show processed image
        rc.display.show_color_image(display_image)
        
        return self.speed, self.angle
    
    def update(self):
        """Update function called every frame."""
        # Get the color image from the camera
        image = rc.camera.get_color_image()
        
        # Ensure image was received
        if image is None:
            print("No image received. Check camera connection.")
            self.control_recovery_mode()
            rc.drive.set_speed_angle(RECOVERY_SPEED, self.angle)
            return
        
        # Get speed and angle from lane follower logic
        speed, angle = self.follow_lanes(image)
        
        # Check for stop signs or other special conditions that might override
        # the regular lane following behavior here
        
        # Send driving commands to the racecar
        rc.drive.set_speed_angle(speed, angle)
        
        # Update the delay to maintain 30 fps
        rc.set_update_slow_time(1/30)
    
    def estimate_lane_curvature(self, lane_points):
        """Estimate the curvature of a lane based on recent points."""
        if len(lane_points) < 3:
            return 0.0  # Not enough points to estimate curvature
            
        # Extract x and y coordinates from lane points
        # Assuming lane_points format is (color_space, x, y)
        x_coords = [point[1] for point in lane_points]
        y_coords = [point[2] for point in lane_points]
        
        # Fit a second-degree polynomial to the points
        if len(x_coords) >= 3 and len(set(y_coords)) >= 2:  # Need at least 3 points and 2 different y values
            try:
                coeffs = np.polyfit(y_coords, x_coords, 2)
                # Curvature is related to the second derivative of the polynomial
                # Here we just use the quadratic coefficient as a simple measure
                return abs(coeffs[0])
            except np.linalg.LinAlgError:
                # Handle case where polyfit fails
                return 0.0
        return 0.0
        
    def detect_obstacles(self):
        """Use LIDAR to detect obstacles in the car's path."""
        samples = rc.lidar.get_samples()
        
        # Check front region for obstacles
        front_dist = rc_utils.get_lidar_average_distance(samples, FRONT_WINDOW[0], FRONT_WINDOW[1])
        
        # Check diagonal regions for obstacles
        front_left_dist = rc_utils.get_lidar_average_distance(samples, -45, -15)
        front_right_dist = rc_utils.get_lidar_average_distance(samples, 15, 45)
        
        # Check side regions
        left_dist = rc_utils.get_lidar_average_distance(samples, LEFT_WINDOW[0], LEFT_WINDOW[1])
        right_dist = rc_utils.get_lidar_average_distance(samples, RIGHT_WINDOW[0], RIGHT_WINDOW[1])
        
        # Return a dictionary of distances
        return {
            'front': front_dist,
            'front_left': front_left_dist,
            'front_right': front_right_dist,
            'left': left_dist,
            'right': right_dist
        }
    
    def handle_tight_turns(self, curvature):
        """Adjust speed and control for tight turns."""
        # Determine if this is a tight turn based on curvature
        if curvature > 0.001:  # Threshold determined experimentally
            # For tight turns, reduce speed further
            tight_turn_factor = 1.0 - min(curvature * 1000, 0.5)  # Cap at 50% reduction
            self.target_speed *= tight_turn_factor
            
            # Optionally adjust steering for tight turns
            # self.angle *= 1.2  # Increase steering angle for sharper response
            
            print(f"Tight turn detected! Curvature: {curvature:.6f}, Speed adjustment: {tight_turn_factor:.2f}")
            return True
        return False

def main():
    """
    Main function, creates and activates the lane follower.
    """
    try:
        print("Alternate Lane Follower starting...")
        
        lane_follower = AdvancedLaneFollower()
        lane_follower.start()
        
        # Set the update function in the racecar
        rc.set_update_slow(lane_follower.update)
        
        # Activate the racecar (this blocks until the program ends)
        rc.go()
        
    except Exception as e:
        print(f"Exception in main: {e}")
        traceback.print_exc()
        
    finally:
        # Make sure the car stops no matter what
        rc.drive.stop()
        print("Alternate Lane Follower stopped.")

if __name__ == "__main__":
    main() 