"""
MIT BWSI Autonomous RACECAR
MIT License
bwsix RC101 - Fall 2023

File Name: lab_i_better.py

Title: Advanced Wall Following with Hybrid Control

Author: [Student Name]

Purpose: Implement a sophisticated wall following algorithm using a hybrid PD and Bang-Bang 
control system. This script allows the car to follow walls at a set distance, using the 
appropriate controller based on current conditions, and provides on-the-fly parameter tuning.

Expected Outcome: The car will follow walls at a consistent distance with minimal oscillation,
handling sharp corners and irregular walls robustly while allowing real-time parameter adjustment.
"""

########################################################################################
# Imports
########################################################################################

import sys
import numpy as np

# Add library directory to the path
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Wall following constants
DESIRED_DISTANCE = 60  # cm
MAX_SPEED = 0.5  # Range from 0 to 1
MIN_SPEED = 0.1  # Lower limit to maintain forward motion

# PD controller constants (initial values, can be tuned with buttons)
KP = 0.5  # Proportional gain
KD = 1.0  # Derivative gain

# Safety threshold to prevent collisions
SAFETY_THRESHOLD = 30  # cm

# Angle ranges for LIDAR scanning
SIDE_ANGLE = 90  # Look directly to the right (3:00 position)
FRONT_ANGLE = 0  # Look directly forward (12:00 position)
SCAN_WINDOW = 20  # Degrees to scan on either side of the main angle

# If wall distance changes by more than this threshold, consider it a corner
CORNER_DETECTION_THRESHOLD = 20  # cm

# Tuning settings
TUNING_ACTIVE = False  # Whether button tuning is active
TUNING_PARAMETER = "KP"  # Which parameter is currently being tuned
TUNING_INCREMENT = 0.1  # How much to change the parameter by

# Variables to store previous measurements for derivative control
prev_error = 0.0
prev_time = 0.0
prev_right_distance = 0.0

# Controller mode
CONTROLLER_MODE = "PD"  # Start with PD controller (alternative is "BANG_BANG")

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    """
    This function is run once every time the start button is pressed
    """
    global prev_error, prev_time, prev_right_distance, TUNING_ACTIVE
    
    # Reset variables
    prev_error = 0.0
    prev_time = 0.0
    prev_right_distance = 0.0
    TUNING_ACTIVE = False
    
    # Print initialization message
    rc_utils.print_colored("Hybrid Wall Following Activated", rc_utils.TerminalColor.green)
    print("Controller Modes:")
    print("- PD Controller: Normal operation")
    print("- Bang-Bang Controller: Sharp corners and irregular walls")
    print("\nUse controller buttons to tune parameters:")
    print("- A: Toggle tuning mode")
    print("- B: Cycle through parameters (KP, KD, DESIRED_DISTANCE)")
    print("- X: Decrease selected parameter")
    print("- Y: Increase selected parameter")
    
    # Start with the car stopped
    rc.drive.stop()


# [FUNCTION] After start() is run, this function is run every frame until the back button
# is pressed
def update():
    """
    Implements hybrid control for wall following
    """
    global prev_error, prev_time, prev_right_distance, KP, KD, DESIRED_DISTANCE
    global TUNING_ACTIVE, TUNING_PARAMETER, CONTROLLER_MODE
    
    # Handle controller input for parameter tuning
    handle_tuning_input()
    
    # Get LIDAR scan data
    scan = rc.lidar.get_samples()
    if scan is None or len(scan) == 0:
        print("WARNING: No LIDAR data received!")
        return
    
    # Get distance to the wall on the right side
    right_angle = SIDE_ANGLE
    right_distance = rc_utils.get_lidar_average_distance(scan, right_angle, SCAN_WINDOW)
    
    # Get the distance to any obstacles in front
    front_distance = rc_utils.get_lidar_average_distance(scan, FRONT_ANGLE, SCAN_WINDOW)
    
    # Find a good point ahead and to the right to see where the wall is going
    ahead_right_angle = 45  # 1:30 position
    ahead_right_distance = rc_utils.get_lidar_average_distance(scan, ahead_right_angle, SCAN_WINDOW)
    
    # Print LIDAR distances for debugging
    if rc.get_delta_time() % 10 == 0:
        print(f"LIDAR distances - Right: {right_distance:.1f} cm, Front: {front_distance:.1f} cm, Ahead-Right: {ahead_right_distance:.1f} cm")
    
    # Calculate error (difference between desired and actual distance to wall)
    error = DESIRED_DISTANCE - right_distance
    
    # Detect if we're approaching a corner
    wall_distance_change = abs(right_distance - prev_right_distance)
    
    # Decide which controller to use
    if wall_distance_change > CORNER_DETECTION_THRESHOLD or abs(error) > DESIRED_DISTANCE * 0.5:
        # Use Bang-Bang controller for sharp corners or large errors
        CONTROLLER_MODE = "BANG_BANG"
        steering_angle = apply_bang_bang_control(error, right_distance, ahead_right_distance)
    else:
        # Use PD controller for normal operation
        CONTROLLER_MODE = "PD"
        steering_angle = apply_pd_control(error, rc.get_delta_time())
    
    # Calculate speed based on front distance
    if front_distance == 0:
        speed = 0.5  # Default speed if no obstacle detected
    else:
        speed = min(0.5, front_distance / 50)  # Scale speed based on front distance
    
    # Apply the calculated control values
    rc.drive.set_speed_angle(speed, steering_angle)
    
    # Store current values for next iteration
    prev_error = error
    prev_right_distance = right_distance
    
    # Print debugging information at a reduced rate (every 5 frames)
    if rc.get_delta_time() % 5 == 0:
        print(f"Mode: {CONTROLLER_MODE}, Right distance: {right_distance:.1f} cm, Error: {error:.1f} cm")
        if CONTROLLER_MODE == "PD":
            print(f"PD Control: P={-KP * error:.2f}")
        print(f"Speed: {speed:.2f}, Angle: {steering_angle:.2f}")
    
    # Display LIDAR visualization
    rc.display.show_lidar(scan)


# [FUNCTION] Apply PD control algorithm and return steering angle
def apply_pd_control(error, dt):
    """
    Calculate steering angle using PD control
    """
    # Calculate derivative of error (rate of change)
    # Only calculate if dt is not too small to avoid division by near-zero
    if dt > 0.001:
        derivative = (error - prev_error) / dt
    else:
        derivative = 0
    
    # Calculate control output using PD control
    # We use negative KP because we want to turn right (negative angle) when error is positive 
    # (when we're too far from the wall)
    control_signal = -KP * error - KD * derivative
    
    # Clamp the control signal to valid steering range (-1 to 1)
    return rc_utils.clamp(control_signal, -1.0, 1.0)


# [FUNCTION] Apply Bang-Bang control algorithm with hysteresis for sharp corners
def apply_bang_bang_control(error, right_distance, ahead_right_distance):
    """
    Calculate steering angle using a modified Bang-Bang controller
    """
    # Two-point Bang-Bang controller as shown in the course images
    
    # Define error thresholds for different actions
    error_threshold_1 = 10  # cm (small error)
    error_threshold_2 = 30  # cm (large error)
    
    # Simple Bang-Bang Control logic with improvements
    if error > error_threshold_2:
        # Far from wall, turn sharply toward it
        return -1.0
    elif error > error_threshold_1:
        # Moderately far from wall, turn gently toward it
        return -0.5
    elif error < -error_threshold_2:
        # Too close to wall, turn sharply away
        return 1.0
    elif error < -error_threshold_1:
        # Moderately close to wall, turn gently away
        return 0.5
    else:
        # Within acceptable range, maintain current heading
        # Look ahead to see if the wall is curving
        if ahead_right_distance > right_distance + 10:
            # Wall is curving away, turn slightly right to follow it
            return -0.3
        elif ahead_right_distance < right_distance - 10:
            # Wall is curving inward, turn slightly left to avoid collision
            return 0.3
        else:
            # Wall is straight ahead, go straight
            return 0.0


# [FUNCTION] Calculate appropriate speed based on conditions
def calculate_speed(steering_angle, front_distance):
    """
    Calculate appropriate speed based on steering angle and front distance
    """
    # Base speed calculation - slow down when turning sharply
    speed = MAX_SPEED - (abs(steering_angle) * 0.5)
    
    # Further reduce speed if there's an obstacle ahead
    if front_distance < 2 * SAFETY_THRESHOLD:
        speed_factor = rc_utils.remap_range(
            front_distance, 
            SAFETY_THRESHOLD, 
            2 * SAFETY_THRESHOLD, 
            MIN_SPEED, 
            speed,
            True
        )
        speed = speed_factor
    
    # Emergency stop if we're about to hit something
    if front_distance < SAFETY_THRESHOLD:
        speed = 0
        rc_utils.print_warning("Obstacle detected ahead! Stopping.")
    
    # Prevent the car from going below minimum speed unless emergency stopping
    if speed > 0:
        speed = max(speed, MIN_SPEED)
    
    # In Bang-Bang mode, slightly reduce speed for stability
    if CONTROLLER_MODE == "BANG_BANG":
        speed = speed * 0.8
    
    return speed


# [FUNCTION] Handle controller button inputs for parameter tuning
def handle_tuning_input():
    """
    Process controller inputs to adjust tuning parameters
    """
    global TUNING_ACTIVE, TUNING_PARAMETER, KP, KD, DESIRED_DISTANCE
    
    # Toggle tuning mode on/off with A button
    if rc.controller.was_pressed(rc.controller.Button.A):
        TUNING_ACTIVE = not TUNING_ACTIVE
        if TUNING_ACTIVE:
            rc_utils.print_colored("Tuning mode ACTIVATED", rc_utils.TerminalColor.green)
        else:
            rc_utils.print_colored("Tuning mode DEACTIVATED", rc_utils.TerminalColor.yellow)
    
    # Only process other buttons if tuning is active
    if TUNING_ACTIVE:
        # Cycle through parameters with B button
        if rc.controller.was_pressed(rc.controller.Button.B):
            if TUNING_PARAMETER == "KP":
                TUNING_PARAMETER = "KD"
            elif TUNING_PARAMETER == "KD":
                TUNING_PARAMETER = "DESIRED_DISTANCE"
            else:
                TUNING_PARAMETER = "KP"
            print(f"Now tuning: {TUNING_PARAMETER}")
        
        # Decrease parameter with X button
        if rc.controller.was_pressed(rc.controller.Button.X):
            if TUNING_PARAMETER == "KP":
                KP = max(0.0, KP - TUNING_INCREMENT)
                print(f"KP decreased to {KP:.1f}")
            elif TUNING_PARAMETER == "KD":
                KD = max(0.0, KD - TUNING_INCREMENT)
                print(f"KD decreased to {KD:.1f}")
            elif TUNING_PARAMETER == "DESIRED_DISTANCE":
                DESIRED_DISTANCE = max(20.0, DESIRED_DISTANCE - 5)
                print(f"DESIRED_DISTANCE decreased to {DESIRED_DISTANCE} cm")
        
        # Increase parameter with Y button
        if rc.controller.was_pressed(rc.controller.Button.Y):
            if TUNING_PARAMETER == "KP":
                KP = min(2.0, KP + TUNING_INCREMENT)
                print(f"KP increased to {KP:.1f}")
            elif TUNING_PARAMETER == "KD":
                KD = min(2.0, KD + TUNING_INCREMENT)
                print(f"KD increased to {KD:.1f}")
            elif TUNING_PARAMETER == "DESIRED_DISTANCE":
                DESIRED_DISTANCE = min(150.0, DESIRED_DISTANCE + 5)
                print(f"DESIRED_DISTANCE increased to {DESIRED_DISTANCE} cm")


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    """
    This function is run at a constant rate that is slower than update().
    """
    # Display controller parameters and status
    print(f"\nController Status:")
    print(f"Mode: {CONTROLLER_MODE}")
    print(f"Parameters: KP: {KP:.2f}, KD: {KD:.2f}")
    print(f"Desired Distance: {DESIRED_DISTANCE} cm")
    print(f"Tuning Active: {TUNING_ACTIVE}, Selected: {TUNING_PARAMETER}")
    

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go() 