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
# Wall Following Specific
DRIVE_SPEED = 1.0
counter_1 = 0.0
DISTANCETRAVELLED = 0.0
Measuring = False

# LIDAR windows (from Hypersonic_TimeTrial.py, latest definitions)
LEFT_WINDOW = (-50, -40) # center : -45
RIGHT_WINDOW = (40, 50) # center : 45
FRONT_WINDOW = (-10, 10)

# General control variables (can be overridden by wall follower or controller)
speed = 0.0
angle = 0.0

# Debug display variables
debug_values = {}

def update_debug_values(key, value):
    """Update the debug_values dictionary with a new key-value pair"""
    debug_values[key] = value

def display_debug_values(image):
    """Display all debug values on the image for wall following"""
    if image is None:
        return image
    
    debug_height = 180  # Increased height for more info
    h, w = image.shape[:2]
    
    # Create a new image with space for debug text, or draw on a copy
    # Ensure we are not modifying the original image if it's used elsewhere
    # For simplicity, let's create the debug overlay area
    display_image = image.copy() # Work on a copy
    
    # Draw a semi-transparent black rectangle at the top for better text visibility
    overlay = display_image.copy()
    cv.rectangle(overlay, (0, 0), (w, debug_height), (0, 0, 0), -1) # Black background
    alpha = 0.6 # Transparency factor
    cv.addWeighted(overlay, alpha, display_image, 1 - alpha, 0, display_image)

    y_offset = 20
    line_height = 20
    font_scale = 0.5
    font_color = (50, 255, 50) # Green
    font_thickness = 1
    font = cv.FONT_HERSHEY_SIMPLEX
    
    debug_info = [
        f"LIDAR Left: {debug_values.get('left_dist', 'N/A'):.1f} cm",
        f"LIDAR Right: {debug_values.get('right_dist', 'N/A'):.1f} cm",
        f"LIDAR Front: {debug_values.get('front_dist', 'N/A'):.1f} cm",
        f"Wall Error: {debug_values.get('wall_error', 'N/A'):.2f}",
        f"Calculated Angle: {debug_values.get('calc_angle', 'N/A'):.2f}",
        f"Calculated Speed: {debug_values.get('calc_speed', 'N/A'):.2f}",
        f"Controller Angle: {debug_values.get('ctrl_angle', 'N/A'):.2f}",
        f"Controller Speed: {debug_values.get('ctrl_speed', 'N/A'):.2f}",
        f"Counter_1: {debug_values.get('counter_1', 'N/A'):.2f} s",
        f"Dist Travelled: {debug_values.get('dist_travelled', 'N/A'):.1f} cm",
        f"Measuring: {debug_values.get('measuring', 'False')}"
    ]

    for i, info_text in enumerate(debug_info):
        cv.putText(
            display_image, info_text, (10, y_offset + i * line_height),
            font, font_scale, font_color, font_thickness
        )
    
    return display_image

def measure_distance_traveled():
    """
    Measures the total distance traveled by the car since the Measuring flag was set to True.
    Uses speed and time to calculate incremental distance.
    
    Returns:
        float: Total distance traveled in cm
    """
    global Measuring, DISTANCETRAVELLED, last_measure_time, speed
    
    current_time_meas = time.time() # Use a different variable name to avoid conflict
    
    if Measuring:
        if 'last_measure_time' not in globals() or last_measure_time == 0.0:
            globals()['last_measure_time'] = current_time_meas
            return DISTANCETRAVELLED
        
        delta_time_meas = current_time_meas - last_measure_time
        
        # Use the car's actual current speed for calculation
        # The global 'speed' might be the target from wall following,
        # or overridden by controller. We need the speed applied to motors.
        # For simplicity, assume 'speed' reflects the command sent to motors.
        # This might need refinement if 'speed' is just a target.
        distance_increment = abs(speed) * delta_time_meas * 100 # Assuming speed is 0-1, 100cm/s max. Uses global speed.
        
        DISTANCETRAVELLED += distance_increment
        globals()['last_measure_time'] = current_time_meas
    else:
        if DISTANCETRAVELLED > 0 or ('last_measure_time' in globals() and last_measure_time > 0):
            DISTANCETRAVELLED = 0.0
            globals()['last_measure_time'] = 0.0
    
    update_debug_values("dist_travelled", DISTANCETRAVELLED)
    update_debug_values("measuring", Measuring)
    return DISTANCETRAVELLED

def update_wall_follower_debug():
    """Updates the wall following behavior with simple proportional control."""
    global counter_1, DISTANCETRAVELLED, Measuring, speed, angle
    
    counter_1 += rc.get_delta_time()
    update_debug_values("counter_1", counter_1)

    if counter_1 < 0.4: # Reset distance if just started or recently reset
        Measuring = False
        measure_distance_traveled() # Resets DISTANCETRAVELLED
    
    Measuring = True # Start measuring
    current_dist_travelled = measure_distance_traveled() # This updates global DISTANCETRAVELLED

    scan = rc.lidar.get_samples()
    
    # Get closest points in window
    # Provide default large values if no points are found
    _, left_dist = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    _, right_dist = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    _, front_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)

    left_dist = left_dist if left_dist > 0 else 1000.0
    right_dist = right_dist if right_dist > 0 else 1000.0
    front_dist = front_dist if front_dist > 0 else 1000.0

    update_debug_values("left_dist", left_dist)
    update_debug_values("right_dist", right_dist)
    update_debug_values("front_dist", front_dist)

    # Display LIDAR
    # rc.display.show_lidar(scan, highlighted_samples=[(rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)[0], left_dist), 
    #                                                 (rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)[0], right_dist), 
    #                                                 (rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)[0], front_dist)])


    error = right_dist - left_dist  
    update_debug_values("wall_error", error)
    
    maxError = 120 # Adjusted from 12 to 120 for cm values
    kP = 0.05 # Adjusted kP, needs tuning

    calc_angle = rc_utils.clamp(kP * error / maxError, -1, 1)
    calc_speed = DRIVE_SPEED

    # Logic from Hypersonic_TimeTrial.py for initial turn boost
    if counter_1 < 3:
        if calc_angle > 0:
            calc_angle = min(1.0, calc_angle + 0.1)
        elif calc_angle < 0:
            calc_angle = max(-1.0, calc_angle - 0.1)
            
    # Update global speed and angle (these can be overridden by controller)
    speed = calc_speed
    angle = calc_angle

    update_debug_values("calc_speed", speed)
    update_debug_values("calc_angle", angle)
    
    return speed, angle


def start():
    """Main start function that initializes wall following debug."""
    global counter_1, DISTANCETRAVELLED, Measuring, speed, angle
    global last_measure_time

    counter_1 = 0.0
    DISTANCETRAVELLED = 0.0
    Measuring = False
    last_measure_time = 0.0 # Initialize for measure_distance_traveled
    speed = 0.0
    angle = 0.0

    rc.drive.stop()
    rc.drive.set_max_speed(1.0) # Allow up to full speed for DRIVE_SPEED
    
    # Initialize debug values
    update_debug_values("left_dist", 0)
    update_debug_values("right_dist", 0)
    update_debug_values("front_dist", 0)
    update_debug_values("wall_error", 0)
    update_debug_values("calc_angle", 0)
    update_debug_values("calc_speed", 0)
    update_debug_values("ctrl_angle", 0)
    update_debug_values("ctrl_speed", 0)
    update_debug_values("counter_1", counter_1)
    update_debug_values("dist_travelled", DISTANCETRAVELLED)
    update_debug_values("measuring", Measuring)
    
    print("Wall Following Debug Tool")
    print("Controls: Left Joystick for steering, Triggers for speed")
    rc.set_update_slow_time(0.5) # For less frequent printing in update_slow

def update():
    """Main update function"""
    global speed, angle # Ensure we modify module-level speed/angle for measure_distance_traveled

    # Run wall following logic. This function:
    # 1. Calls measure_distance_traveled(), which uses the global 'speed' from the *previous* frame's end.
    # 2. Calculates its own 'calc_speed' and 'calc_angle'.
    # 3. Sets the global 'speed' and 'angle' internally to these 'calc_speed' and 'calc_angle'.
    # 4. Updates debug_values for 'calc_speed', 'calc_angle'.
    # The return values are not strictly needed here as the function has side effects on globals and debug_values.
    update_wall_follower_debug()
    # At this point, global 'speed' and 'angle' reflect the wall follower's calculated values.

    # Get controller inputs
    joystick_vals = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    controller_final_angle = joystick_vals[0]  # X-axis for angle
    
    right_trigger = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    left_trigger = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    controller_final_speed = right_trigger - left_trigger
    
    update_debug_values("ctrl_speed", controller_final_speed)
    update_debug_values("ctrl_angle", controller_final_angle)

    # Drive the car using ONLY controller inputs
    rc.drive.set_speed_angle(controller_final_speed, controller_final_angle)

    # NOW, update the global 'speed' and 'angle' to what was ACTUALLY sent to the motors.
    # This ensures that the 'speed' value used by measure_distance_traveled() in the
    # *next* call to update_wall_follower_debug() is the speed commanded by the controller
    # in the current frame.
    speed = controller_final_speed
    angle = controller_final_angle
            
    # Display camera with debug info
    image = rc.camera.get_color_image()
    if image is not None:
        debug_image = display_debug_values(image)
        rc.display.show_color_image(debug_image)
    
    # Display LIDAR (can be intensive, enable if needed)
    scan = rc.lidar.get_samples()
    if scan is not None:
         # Get raw closest points for highlighting
        raw_left_angle, raw_left_dist = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
        raw_right_angle, raw_right_dist = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
        raw_front_angle, raw_front_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)

        highlighted_samples = []
        if raw_left_dist > 0: highlighted_samples.append((raw_left_angle, raw_left_dist))
        if raw_right_dist > 0: highlighted_samples.append((raw_right_angle, raw_right_dist))
        if raw_front_dist > 0: highlighted_samples.append((raw_front_angle, raw_front_dist))
        
        rc.display.show_lidar(scan, highlighted_samples=highlighted_samples)


def update_slow():
    """Slow update function for less critical tasks, like printing."""
    # print(f"Counter: {counter_1:.2f}, Dist: {DISTANCETRAVELLED:.2f}, Speed: {speed:.2f}, Angle: {angle:.2f}")
    # This can be used for printing less frequently than the main update loop.
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go() 