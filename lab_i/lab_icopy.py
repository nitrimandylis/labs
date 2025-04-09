"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020
Lab 4B - LIDAR Wall Following
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
# Removed the simple_pid import and implementing PID control directly

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here
class State(IntEnum):
    forward = 0
    turn = 1

cur_state = State.forward
speed = 0.0
angle = 0.0

# PID controller constants
kp = 0.015
ki = 0.1
kd = 0.05

# Previous error for derivative term
prev_error = 0.0
# Integral accumulator
error_sum = 0.0
# Maximum integral windup
MAX_ERROR_SUM = 100.0

# Speed control constants
MAX_SPEED = 1.0
MIN_SPEED = 0.1
SAFETY_DISTANCE = 50  # cm

########################################################################################
# Functions
########################################################################################

def apply_pid_control(error, prev_error, error_sum, dt=1.0):
    """
    Calculate control output using PID control
    """
    # Proportional term
    p_term = kp * error
    
    # Integral term with anti-windup
    error_sum += error * dt
    error_sum = max(-MAX_ERROR_SUM, min(error_sum, MAX_ERROR_SUM))  # Anti-windup
    i_term = ki * error_sum
    
    # Derivative term
    d_term = kd * (error - prev_error) / dt if dt > 0.001 else 0
    
    # Calculate control output
    control = p_term + i_term + d_term
    
    # Clamp the control signal to valid range
    return control, error_sum


def calculate_speed(forward_distance):
    """
    Calculate appropriate speed based on forward distance
    """
    if forward_distance == 0:
        return MAX_SPEED  # Default speed if no obstacle detected
    
    # Scale speed based on forward distance
    speed = rc_utils.remap_range(
        forward_distance,
        0,
        SAFETY_DISTANCE,
        MIN_SPEED,
        MAX_SPEED,
        True
    )
    
    # Ensure speed stays within valid range
    return rc_utils.clamp(speed, MIN_SPEED, MAX_SPEED)


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()
    global cur_state
    global speed
    global angle
    global prev_error
    global error_sum
    
    # Reset PID controller variables
    prev_error = 0.0
    error_sum = 0.0
    
    # Print start message
    print(">> Lab 4B - LIDAR Wall Following")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global cur_state
    global speed
    global angle
    global prev_error
    global error_sum
    
    # TODO: Follow the wall to the right of the car without hitting anything.
    scan = rc.lidar.get_samples()
    
    forward_distance = rc_utils.get_lidar_average_distance(scan, 0, 10)
    top_right = rc_utils.get_lidar_average_distance(scan, 42, 10)
    top_left = rc_utils.get_lidar_average_distance(scan, 318, 10)

    diff_top = top_right - top_left 
    diff_top2 = top_left - top_right
    
    # Calculate PID control output
    control, error_sum = apply_pid_control(diff_top2, prev_error, error_sum, rc.get_delta_time())
    # Store current error for next iteration
    prev_error = diff_top2

    # Calculate speed using the new function
    speed = calculate_speed(forward_distance)
    
    angle = rc_utils.clamp(control, -1.55, 1.55)
    
    if (abs(angle) < 0.05):
        angle = 0 
    speed = min(max(-1, speed), 1)
    angle = min(max(-1, angle), 1)
    rc.drive.set_speed_angle(speed, -angle)
    
    # Display LIDAR visualization
    rc.display.show_lidar(scan)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()