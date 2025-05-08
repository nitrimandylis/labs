import sys
import os
import cv2 as cv
import numpy as np
import math
import time

# Add racecar library to path
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

# Initialize racecar
rc = racecar_core.create_racecar()

# Global variables
accumulatedError = 0
lastError = 0
contour_center = (0, 0)
contour_area = 0
largestcontour_center = (0, 0)  
secondcontour_center = (0, 0)
generalcontour_center = (0, 0)
current_time = 0
speed = 0
angle = 0
recovery_mode = False
recovery_counter = 0
last_good_angle = 0
previous_centers = []
MAX_SPEED = 0.5
Lane_priority = 0  # 0 for PURPLE, 1 for ORANGE

# Color ranges
ORANGE = ((10, 100, 100), (20, 255, 255))
PURPLE = ((130, 60, 60), (150, 255, 255))
prioritylist = [PURPLE]  # Default
MIN_CONTOUR_AREA = 30

# LIDAR windows
LEFT_WINDOW = (-135, -45)
RIGHT_WINDOW = (45, 135)
FRONT_WINDOW = (-10, 10)

# Debug display variables
debug_values = {}

def update_debug_values(key, value):
    """Update the debug_values dictionary with a new key-value pair"""
    debug_values[key] = value

def display_debug_values(image):
    """Display all debug values on the image"""
    if image is None:
        return image
    
    # Create a blank area at the top of the image for text
    debug_height = 150  # Height of debug area
    h, w = image.shape[:2]
    debug_image = np.zeros((h + debug_height, w, 3), dtype=np.uint8)
    debug_image[debug_height:, :, :] = image
    
    # Background for text area
    cv.rectangle(debug_image, (0, 0), (w, debug_height), (30, 30, 30), -1)
    
    # Display contour information
    y_offset = 30
    line_height = 30
    font_scale = 0.7
    font_color = (50, 255, 50)
    font_thickness = 2
    font = cv.FONT_HERSHEY_SIMPLEX
    
    # Left contour info
    if debug_values.get('largest_contour_visible', False):
        # Distance text
        cv.putText(
            debug_image,
            f"LEFT CONTOUR DISTANCE: {debug_values.get('largest_contour_distance', 0):.1f} pixels",
            (20, y_offset),
            font, font_scale, font_color, font_thickness
        )
        
        # Angle text
        cv.putText(
            debug_image,
            f"LEFT CONTOUR ANGLE: {debug_values.get('largest_contour_angle', 0):.2f} degrees",
            (20, y_offset + line_height),
            font, font_scale, font_color, font_thickness
        )
    else:
        cv.putText(
            debug_image,
            "LEFT CONTOUR: NOT VISIBLE",
            (20, y_offset),
            font, font_scale, (50, 50, 255), font_thickness
        )
    
    # Right contour info
    if debug_values.get('second_contour_visible', False):
        # Distance text
        cv.putText(
            debug_image,
            f"RIGHT CONTOUR DISTANCE: {debug_values.get('second_contour_distance', 0):.1f} pixels",
            (20, y_offset + line_height * 2),
            font, font_scale, font_color, font_thickness
        )
        
        # Angle text
        cv.putText(
            debug_image,
            f"RIGHT CONTOUR ANGLE: {debug_values.get('second_contour_angle', 0):.2f} degrees",
            (20, y_offset + line_height * 3),
            font, font_scale, font_color, font_thickness
        )
    else:
        cv.putText(
            debug_image,
            "RIGHT CONTOUR: NOT VISIBLE",
            (20, y_offset + line_height * 2),
            font, font_scale, (50, 50, 255), font_thickness
        )
    
    return debug_image

def pid_Lane(Kp, Ki, Kd, target, current, dT):
    global accumulatedError
    global lastError
    error = target - current
    accumulatedError += error * dT
    accumulatedError = rc_utils.clamp(accumulatedError, -10, 10)
    deltaError = (error - lastError) / dT if dT > 0 else 0
    pTerm = Kp * error
    iTerm = Ki * accumulatedError
    dTerm = Kd * deltaError
    lastError = error
    
    pid_output = pTerm + iTerm + dTerm
    update_debug_values("pid_output", pid_output)
    update_debug_values("pid_error", error)
    
    return pid_output

def update_contour_Lane():
    global contour_center, contour_area
    global largestcontour_center, secondcontour_center, generalcontour_center
    global current_time

    image = rc.camera.get_color_image()
    if image is None:
        return (0, 0), (0, 0), (0, 0)
    
    # Get camera dimensions for calculations
    camera_width = rc.camera.get_width()
    camera_height = rc.camera.get_height()
    image_center_x = camera_width // 2
    
    crop_floor = ((0, 0), (rc.camera.get_height(), rc.camera.get_width()))
    image = rc_utils.crop(image, crop_floor[0], crop_floor[1])
    
    largestcontour = None
    secondcontour = None
    largestcontour_center = (0, 0)
    secondcontour_center = (0, 0)
    generalcontour_center = (0, 0)
    
    for col in prioritylist:
        contours = rc_utils.find_contours(image, col[0], col[1])
        
        if not contours or len(contours) == 0:
            continue
            
        valid_contours = []
        for contour in contours:
            if contour is not None and cv.contourArea(contour) > MIN_CONTOUR_AREA:
                valid_contours.append(contour)
        
        if len(valid_contours) == 0:
            continue
        
        sorted_contours = sorted(valid_contours, key=cv.contourArea, reverse=True)
        
        largestcontour = sorted_contours[0] if len(sorted_contours) > 0 else None
        secondcontour = sorted_contours[1] if len(sorted_contours) > 1 else None
        
        if largestcontour is not None:
            break
    
    # Reset visibility flags
    update_debug_values("largest_contour_visible", False)
    update_debug_values("second_contour_visible", False)
    
    # Process largest contour (left line)
    if largestcontour is not None:
        largestcontour_center = rc_utils.get_contour_center(largestcontour)
        rc_utils.draw_contour(image, largestcontour, (0, 255, 0))
        
        if largestcontour_center is not None:
            # Draw circle on contour center
            rc_utils.draw_circle(image, largestcontour_center, (0, 255, 0))
            
            # Calculate distance and angle to contour
            x, y = largestcontour_center
            
            # Distance is straightforward Euclidean distance
            # From bottom center of image to contour center
            bottom_center_y = image.shape[0]  # Bottom of the image after cropping
            distance_pixels = np.sqrt((x - image_center_x)**2 + (y - bottom_center_y)**2)
            
            # Angle calculation: 0 is straight ahead, positive is right, negative is left
            # First normalize x position to be between -1 and 1
            normalized_x = (x - image_center_x) / (image_center_x)
            # Convert to degrees (roughly)
            angle_degrees = normalized_x * 45  # Assuming ~45 degrees field of view in half the image
            
            # Store values for display
            update_debug_values("largest_contour_visible", True)
            update_debug_values("largest_contour_distance", distance_pixels)
            update_debug_values("largest_contour_angle", angle_degrees)
            
            generalcontour_center = largestcontour_center
            contour_center = largestcontour_center
            contour_area = cv.contourArea(largestcontour)
    
    # Process second contour (right line)
    if secondcontour is not None:
        secondcontour_center = rc_utils.get_contour_center(secondcontour)
        rc_utils.draw_contour(image, secondcontour, (255, 0, 0))
        
        if secondcontour_center is not None:
            # Draw circle on contour center
            rc_utils.draw_circle(image, secondcontour_center, (255, 0, 0))
            
            # Calculate distance and angle to contour
            x, y = secondcontour_center
            
            # Distance calculation
            bottom_center_y = image.shape[0]  # Bottom of the image after cropping
            distance_pixels = np.sqrt((x - image_center_x)**2 + (y - bottom_center_y)**2)
            
            # Angle calculation
            normalized_x = (x - image_center_x) / (image_center_x)
            angle_degrees = normalized_x * 45
            
            # Store values for display
            update_debug_values("second_contour_visible", True)
            update_debug_values("second_contour_distance", distance_pixels)
            update_debug_values("second_contour_angle", angle_degrees)

    # Add debug display to image
    debug_image = display_debug_values(image)
    rc.display.show_color_image(debug_image)
    
    current_time += rc.get_delta_time()
    return largestcontour_center, secondcontour_center, generalcontour_center

def follow_two_lines_Lane():
    global speed, angle, last_good_angle, recovery_mode, recovery_counter, previous_centers
    
    cameraWidth = rc.camera.get_width()
    distancethreshold = 70
    delta_time = rc.get_delta_time()
    
    largestcontour_center, secondcontour_center, generalcontour_center = update_contour_Lane()
    
    has_largest = isinstance(largestcontour_center, tuple) and len(largestcontour_center) == 2 and largestcontour_center[0] != 0
    has_second = isinstance(secondcontour_center, tuple) and len(secondcontour_center) == 2 and secondcontour_center[0] != 0
    has_general = isinstance(generalcontour_center, tuple) and len(generalcontour_center) == 2 and generalcontour_center[0] != 0
    
    update_debug_values("has_largest", has_largest)
    update_debug_values("has_second", has_second)
    
    if has_largest and has_second:
        smallestx = min(largestcontour_center[1], secondcontour_center[1])
        largestx = max(largestcontour_center[1], secondcontour_center[1])
        center_point = (largestx + smallestx) / 2
        
        lane_width = largestx - smallestx
        update_debug_values("lane_width", lane_width)
        
        if (largestx - smallestx) > distancethreshold:
            target_point = center_point
            normalized_target = rc_utils.remap_range(target_point, 0, cameraWidth, -1, 1)
            angle = pid_Lane(0.5, 0.1, 0.2, 0, normalized_target, delta_time)
            speed = 1
            update_debug_values("control_mode", "PID")
        else:
            if center_point < (cameraWidth/2) - 30:
                angle = rc_utils.remap_range(center_point, 0, cameraWidth/2, 0.5, 0.1)
            elif center_point > (cameraWidth/2) + 30:
                angle = rc_utils.remap_range(center_point, cameraWidth/2, cameraWidth, -0.1, -0.5)
            speed = 1
            update_debug_values("control_mode", "Direct")
        
        previous_centers.append((largestcontour_center[0], center_point))
        if len(previous_centers) > 4:
            previous_centers.pop(0)
            
        basic_angle = center_point - cameraWidth / 2
        basic_angle /= cameraWidth / 2
        last_good_angle = basic_angle
        
        recovery_mode = False
        recovery_counter = 0
        update_debug_values("center_point", center_point)
            
    elif has_general:
        center_point = generalcontour_center[1]
        update_debug_values("center_point", center_point)
        update_debug_values("control_mode", "Single Line")
        
        if center_point < cameraWidth/2:
            angle = rc_utils.remap_range(center_point, 0, cameraWidth/2, 0.5, 0.1)
        else:
            angle = rc_utils.remap_range(center_point, cameraWidth/2, cameraWidth, -0.1, -0.5)
        
        speed = 1
        
        previous_centers.append((generalcontour_center[0], center_point))
        if len(previous_centers) > 4:
            previous_centers.pop(0)
            
        basic_angle = center_point - cameraWidth / 2
        basic_angle /= cameraWidth / 2
        last_good_angle = basic_angle
        
        recovery_mode = False
        recovery_counter = 0
        
    else:
        enter_recovery_mode_Lane(delta_time)
    
    update_debug_values("speed", speed)
    update_debug_values("angle", angle)
    update_debug_values("recovery_mode", recovery_mode)

def enter_recovery_mode_Lane(delta_time):
    global recovery_mode, recovery_counter, angle, speed, previous_centers
    
    if not recovery_mode:
        recovery_mode = True
        recovery_counter = 0
        update_debug_values("control_mode", "RECOVERY")
    
    if len(previous_centers) > 0:
        if len(previous_centers) > 1:
            previous_centers.pop(0)
    
    recovery_counter += delta_time
    update_debug_values("recovery_counter", recovery_counter)
    
    angle = last_good_angle
    
    search_amplitude = 0.3
    search_frequency = 1.5
    search_offset = math.sin(recovery_counter * search_frequency * 2 * math.pi) * search_amplitude
    angle += search_offset
    update_debug_values("search_offset", search_offset)
    
    speed = 1
    
    if recovery_counter > 2.0:
        recovery_counter = 0

def start_Lane():
    global recovery_mode, recovery_counter, last_good_angle, previous_centers
    global accumulatedError, lastError, speed, angle, current_time, MAX_SPEED
    
    accumulatedError = 0
    lastError = 0
    current_time = 0
    recovery_mode = False
    recovery_counter = 0
    last_good_angle = 0
    previous_centers = []
    speed = 0
    angle = 0
    
    MAX_SPEED = 0.5
    rc.drive.set_speed_angle(0, 0)
    rc.drive.set_max_speed(MAX_SPEED)
    
    print(
        ">> Lab F - Two Line Following Challenge\n"
        "\n"
        "Controls:\n"
        "   X button = set ORANGE as primary color\n"
        "   Y button = set PURPLE as primary color"
    )
    
    rc.set_update_slow_time(0.5)

def update_Lane():
    global prioritylist, angle, speed, Lane_priority, MAX_SPEED
    
    # Set priority based on Lane_priority value
    if Lane_priority == 1:
        prioritylist = [ORANGE]
    else:  # Lane_priority == 0
        prioritylist = [PURPLE]
    
    update_debug_values("lane_priority", Lane_priority)
    
    # Get LIDAR scan for wall detection
    scan = rc.lidar.get_samples()
    left_angle, left_distance = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    front_angle, front_distance = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    
    # Default values if no point detected
    left_distance = 1000 if left_distance is None else left_distance
    right_distance = 1000 if right_distance is None else right_distance
    front_distance = 1000 if front_distance is None else front_distance
    
    # Update debug values for LIDAR
    update_debug_values("left_dist", left_distance)
    update_debug_values("right_dist", right_distance)
    update_debug_values("front_dist", front_distance)
    
    # Follow lanes first
    follow_two_lines_Lane()
    
    # Wall avoidance logic
    wall_avoidance_angle = 0
    
    # If wall is too close on the left, turn right
    if left_distance < 60:
        wall_avoidance_angle = rc_utils.remap_range(left_distance, 20, 60, 0.7, 0.2)
        update_debug_values("wall_side", "LEFT")
    
    # If wall is too close on the right, turn left
    elif right_distance < 60:
        wall_avoidance_angle = rc_utils.remap_range(right_distance, 20, 60, -0.7, -0.2)
        update_debug_values("wall_side", "RIGHT")
    
    # If wall is directly in front, make a stronger turn
    if front_distance < 100:
        # Turn away from the closest side wall, or choose left by default if equal
        if left_distance < right_distance:
            wall_avoidance_angle = 0.8  # Turn right
            update_debug_values("wall_side", "FRONT-RIGHT")
        else:
            wall_avoidance_angle = -0.8  # Turn left
            update_debug_values("wall_side", "FRONT-LEFT")
        
        # Slow down when approaching a wall
        speed = rc_utils.remap_range(front_distance, 30, 100, 0.5, 1.0)
    
    update_debug_values("wall_angle", wall_avoidance_angle)
    
    # Combine lane following angle with wall avoidance
    # Lane following gets priority, but wall avoidance can override if needed
    if abs(wall_avoidance_angle) > 0.1:
        # Blend the angles, with more weight to wall avoidance when walls are very close
        blend_factor = rc_utils.remap_range(min(left_distance, right_distance, front_distance), 
                                          20, 60, 0.8, 0.3)
        blend_factor = rc_utils.clamp(blend_factor, 0, 0.8)
        
        # Apply blending
        original_angle = angle
        angle = angle * (1 - blend_factor) + wall_avoidance_angle * blend_factor
        update_debug_values("blend_factor", blend_factor)
        update_debug_values("original_angle", original_angle)
    
    # Apply additional steering bias for sharper turns
    if angle > 0:
        angle += 0.4
    elif angle < 0:
        angle -= 0.5
  
    angle = rc_utils.clamp(angle, -1, 1)
    
    speed_factor = 1.0 - abs(angle) * 1.5
    calculate_speed = speed * max(0.5, speed_factor)
    rc.drive.set_max_speed(MAX_SPEED)
    calculate_speed = 1
    
    update_debug_values("speed", calculate_speed)
    update_debug_values("angle", angle)
    
    return calculate_speed, angle

def start():
    """Main start function that initializes lane following"""
    global Lane_priority
    
    Lane_priority = 0  # Default to PURPLE
    start_Lane()
    
    # Initialize debug values
    update_debug_values("speed", 0)
    update_debug_values("angle", 0)
    update_debug_values("recovery_mode", False)
    update_debug_values("lane_priority", Lane_priority)
    
    rc.set_update_slow_time(1.0)

def update():
    """Main update function that handles controller input and calls lane following update"""
    global Lane_priority
    
    # Handle controller inputs for lane color selection
    if rc.controller.was_pressed(rc.controller.Button.X):
        Lane_priority = 1  # ORANGE
        print("Lane color set to ORANGE")
    
    if rc.controller.was_pressed(rc.controller.Button.Y):
        Lane_priority = 0  # PURPLE
        print("Lane color set to PURPLE")
    
    # Update lane following and apply speed/angle
    speed, angle = update_Lane()
    rc.drive.set_speed_angle(speed, angle)
    joystick_vals = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    controller_angle = joystick_vals[0]  # Extract x component
    
    # Right trigger for forward, left trigger for reverse
    right_trigger = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    left_trigger = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    controller_speed = right_trigger - left_trigger
    
    # Always apply user's controller inputs to car
    rc.drive.set_speed_angle(controller_speed, controller_angle)
    

def update_slow():
    """Slow update function for less critical tasks"""
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go() 