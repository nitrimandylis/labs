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
from typing import Tuple

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Wall following constants
WALL_DISTANCE = 50  # Desired distance from wall in cm
MAX_SPEED = 0.8     # Maximum speed of the car
TURN_SPEED = 0.5    # Speed while turning
SCAN_WINDOW = 20    # Width of LIDAR scan window in degrees

# LIDAR angles
RIGHT_SIDE = 60     # Angle for right wall (60° = 2 o'clock position)
LEFT_SIDE = 300     # Angle for left wall (300° = 10 o'clock position)

########################################################################################
# Functions
########################################################################################

def get_wall_distances() -> Tuple[float, float]:
    """
    Get the distances to walls on both sides of the car using LIDAR
    Returns: (left_distance, right_distance)
    """
    scan = rc.lidar.get_samples()
    if scan is None:
        return 0, 0
        
    right_dist = rc_utils.get_lidar_average_distance(scan, RIGHT_SIDE, SCAN_WINDOW)
    left_dist = rc_utils.get_lidar_average_distance(scan, LEFT_SIDE, SCAN_WINDOW)
    
    return left_dist, right_dist


def calculate_steering_angle(left_dist: float, right_dist: float) -> float:
    """
    Calculate steering angle based on the difference between left and right wall distances
    Returns: angle between -1 (full left) and 1 (full right)
    """
    # Calculate the error (difference between left and right distances)
    difference = right_dist - left_dist
    
    # Map the difference to a steering angle
    # If difference is positive (right wall further), turn right
    # If difference is negative (left wall further), turn left
    raw_angle = rc_utils.remap_range(difference, -15, 15, -1, 1)
    
    # Clamp the angle to valid range
    return rc_utils.clamp(raw_angle, -1, 1)


def show_lidar_points():
    scan = rc.lidar.get_samples()
    if scan is None:
        return
        
    # Get the points we're using for wall tracking
    right_points = []
    left_points = []
    
    # Calculate angle ranges
    right_start = RIGHT_SIDE - SCAN_WINDOW // 2
    right_end = RIGHT_SIDE + SCAN_WINDOW // 2
    left_start = LEFT_SIDE - SCAN_WINDOW // 2
    left_end = LEFT_SIDE + SCAN_WINDOW // 2
    
    # Collect points in the scan windows
    for angle in range(right_start, right_end):
        if 0 < scan[angle] < 1000:  # Only include valid points
            # Convert array index to LIDAR angle (clockwise from front)
            lidar_angle = (angle * 360) // len(scan)
            right_points.append((lidar_angle, scan[angle]))
            
    for angle in range(left_start, left_end):
        if 0 < scan[angle] < 1000:  # Only include valid points
            # Convert array index to LIDAR angle (clockwise from front)
            lidar_angle = (angle * 360) // len(scan)
            left_points.append((lidar_angle, scan[angle]))
    
    # Combine all points to highlight
    highlighted_points = right_points + left_points
    
    rc.display.create_window()
    rc.display.show_lidar(scan, radius=512, max_range=1000, highlighted_samples=highlighted_points)


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Start with the car stopped
    rc.drive.set_max_speed(0.2)
    rc.drive.stop()


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Get LIDAR scan
    scan = rc.lidar.get_samples()
    if scan is None:
        return
    
    # Get distances to walls on both sides
    left_dist, right_dist = get_wall_distances()
    
    # Calculate appropriate steering angle
    angle = calculate_steering_angle(left_dist, right_dist)
    
    # Use max speed since speed control is handled by the angle calculation
    speed = MAX_SPEED
    print("left_dist: ", left_dist, "right_dist: ", right_dist)
    if right_dist > 150 and left_dist > 150:
        rc.drive.set_max_speed(1)
    elif right_dist > 80 and left_dist > 80:
        rc.drive.set_max_speed(0.5)
    elif right_dist < 80 and left_dist < 80:
        rc.drive.set_max_speed(0.2)
    # Set the speed and angle
    rc.drive.set_speed_angle(1, angle)
    
    show_lidar_points()


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
    #70