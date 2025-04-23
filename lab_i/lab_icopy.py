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
FRONT_CENTER = 0    # Angle for front detection (0° = directly ahead)

# Obstacle avoidance constants
FRONT_SCAN_ANGLE = 60  # Total width of front scanning window in degrees
OBSTACLE_THRESHOLD =  70  # Distance in cm to consider an obstacle dangerous

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


def check_front_obstacle() -> Tuple[bool, float, float]:
    """
    Check for obstacles in front of the car
    Returns: (obstacle_detected, min_distance, angle_to_obstacle)
    """
    scan = rc.lidar.get_samples()
    if scan is None:
        return False, 0, 0
    
    # Get front scan area (from -FRONT_SCAN_ANGLE/2 to +FRONT_SCAN_ANGLE/2)
    front_scan_start = len(scan) // 2 - FRONT_SCAN_ANGLE // 2
    front_scan_end = len(scan) // 2 + FRONT_SCAN_ANGLE // 2
    
    # Find minimum distance in front area
    min_dist = float('inf')
    min_angle_index = 0
    
    for i in range(front_scan_start, front_scan_end):
        # Use modulo to handle wrap-around at the end of the array
        idx = i % len(scan)
        if 0 < scan[idx] < min_dist:
            min_dist = scan[idx]
            min_angle_index = idx
    
    # Convert index to angle (-180 to 180 degrees)
    angle = ((min_angle_index * 360) // len(scan)) % 360
    if angle > 180:
        angle -= 360  # Convert to -180 to 180 range
    
    # Check if obstacle is detected (distance below threshold)
    obstacle_detected = min_dist < OBSTACLE_THRESHOLD
    
    return obstacle_detected, min_dist, angle


def find_clear_path() -> float:
    """
    Find the clearest path around obstacles
    Returns: steering angle (-1 to 1) toward the clearest path
    """
    scan = rc.lidar.get_samples()
    if scan is None:
        return 0
    
    # Search a wider area to find the best path
    search_width = 150  # degrees
    search_start = len(scan) // 2 - search_width // 2
    search_end = len(scan) // 2 + search_width // 2
    
    # Find the farthest point (clearest path)
    max_dist = 0
    best_angle_index = len(scan) // 2  # Default to straight ahead
    
    for i in range(search_start, search_end):
        idx = i % len(scan)
        if scan[idx] > max_dist:
            max_dist = scan[idx]
            best_angle_index = idx
    
    # Convert index to angle (-180 to 180 degrees)
    angle = ((best_angle_index * 360) // len(scan)) % 360
    if angle > 180:
        angle -= 360
    
    # Map angle to steering range (-1 to 1)
    steering = rc_utils.remap_range(angle, -60, 60, -1, 1)
    return rc_utils.clamp(steering, -1, 1)


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
    front_points = []
    
    # Calculate angle ranges
    right_start = RIGHT_SIDE - SCAN_WINDOW // 2
    right_end = RIGHT_SIDE + SCAN_WINDOW // 2
    left_start = LEFT_SIDE - SCAN_WINDOW // 2
    left_end = LEFT_SIDE + SCAN_WINDOW // 2
    front_start = len(scan) // 2 - FRONT_SCAN_ANGLE // 2
    front_end = len(scan) // 2 + FRONT_SCAN_ANGLE // 2
    
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

    # Add front scan points
    for i in range(front_start, front_end):
        idx = i % len(scan)
        if 0 < scan[idx] < 1000:
            lidar_angle = (idx * 360) // len(scan)
            front_points.append((lidar_angle, scan[idx]))
    
    # Combine all points to highlight
    highlighted_points = right_points + left_points + front_points
    
    rc.display.create_window()
    rc.display.show_lidar(scan, radius=512, max_range=1000, highlighted_samples=highlighted_points)


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Start with the car stopped
    rc.drive.set_max_speed(0.2)
    global counter
    counter = 0
    global angle
    angle = 0
    global speed
    speed = 1
    rc.drive.stop()


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Get LIDAR scan

    global counter, angle, speed

    scan = rc.lidar.get_samples()
    if scan is None:
        return
    
    # First check for obstacles in front
    obstacle_detected, obstacle_distance, obstacle_angle = check_front_obstacle()
    
    # Get distances to walls on both sides
    left_dist, right_dist = get_wall_distances()
    
    # Determine speed based on environment
    if right_dist > 150 and left_dist > 150:
        rc.drive.set_max_speed(1)
    elif right_dist > 80 and left_dist > 80:
        rc.drive.set_max_speed(0.5)
    elif right_dist < 80 and left_dist < 80:
        rc.drive.set_max_speed(0.2)
    
    # Calculate steering angle
    if obstacle_detected:
        # If obstacle is detected, find clear path around it
        angle = find_clear_path()
        counter += rc.get_delta_time()
        # Reduce speed when avoiding obstacles
        speed = rc_utils.remap_range(obstacle_distance, 0, OBSTACLE_THRESHOLD, 0.2, 0.5)
        speed = 1
        
        print(f"OBSTACLE DETECTED! Distance: {obstacle_distance:.1f}cm, Steering: {angle:.2f}")
    elif counter > 3 or not obstacle_detected and counter == 0:
        # No obstacles, use normal wall following
        
        angle = calculate_steering_angle(left_dist, right_dist)
        speed = 1.0
        print("Wall following - left_dist: ", left_dist, "right_dist: ", right_dist)
    
    # Set the speed and angle
    print(counter)
    rc.drive.set_speed_angle(speed, angle)
    
    show_lidar_points()


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
    #70