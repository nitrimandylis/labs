"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Wall Following Module - extracted from Grand Prix Driver
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import math
import random
import time

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum
from enum import Enum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# LIDAR window definitions
LEFT_WINDOW = (-45, -15)
RIGHT_WINDOW = (15, 45)
FRONT_WINDOW = (-10, 10)  # Define a window directly in front of the car

# State definition for wall following
class State(IntEnum):
    move = 0
    turn = 1
    stop = 2

# Initialize state
cur_state = State.move

# LIDAR variables
speed = 0.0
angle = 0.0
left_angle = 0
left_distance = 0
right_angle = 0
right_distance = 0
front_distance = 1000
random_number = 1

# Yellow detection
angle_to_yellow = 0  # Angle to the detected yellow object (-1.0 to 1.0)
YELLOW = ((20, 100, 100), (40, 255, 255))  # HSV range for yellow

########################################################################################
# Functions
########################################################################################

def stop_wall_following():
    """
    Stop the car when an obstacle is detected directly in front
    """
    global speed
    global angle
    global cur_state
    global front_distance

    speed = 0
    angle = 0
    
    # If the path is clear again, start moving
    if front_distance > 40:
        cur_state = State.move


def turn_wall_following():
    """
    Turn the car to balance distances to walls on both sides
    """
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global front_distance
    global angle
    global cur_state
    global speed

    error = right_distance - left_distance
    if error < 0:
        # Turn left - simplify angle calculation 
        angle = rc_utils.remap_range(error, -15, 15, -1, 1)
        
        # Apply fixed adjustments to make turns more responsive
        # if -0.4 < angle < 0:
        #     angle -= 0.1
        # elif angle < -0.4:
        #     angle -= 0.4
    else:   
        # Turn right - simplify angle calculation
        angle = rc_utils.remap_range(error, -15, 15, -1, 1)
        
        # Apply fixed adjustments to make turns more responsive
        # if 0 < angle < 0.4:
        #     angle += 0.1
        # elif angle > 0.4:
        #     angle += 0.4
            
    # Make sure angle stays within bounds
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    
    # Set appropriate speed for turning
    rc.drive.set_max_speed(0.28)

    # Override with stronger turn if obstacle directly ahead
    if front_distance < 30 and left_distance == right_distance:
        angle = 1

    # Exit turn state if walls are balanced
    if abs(error) < 10:
        cur_state = State.move
    
    speed = 1

    Norm_angle = rc_utils.remap_range(error, -15, 15, -1, 1)

    return rc_utils.clamp(Norm_angle, -1, 1)

def move_wall_following():
    """
    Move forward while walls are balanced
    """
    global speed
    global angle
    global left_distance
    global right_distance
    global front_distance
    global cur_state

    # Set default values for moving forward
    speed = 1
    angle = 0
    
    # Check if we need to turn based on wall distances
    if abs(left_distance-right_distance) > 10:
        cur_state = State.turn

def can_see_yellow(min_contour_area=100):
    """
    Detect yellow objects (e.g., tennis balls) in the camera view
    """
    global angle_to_yellow
    
    # Capture image once and check if it's valid
    image = rc.camera.get_color_image()
    if image is None:
        return False, None, 0
    
    # Create one focused crop region
    crop_bottom = int(rc.camera.get_height() * 2/3)
    crop_region = ((crop_bottom, 0), (rc.camera.get_height(), rc.camera.get_width()))
    
    # Crop the image
    cropped_image = rc_utils.crop(image, crop_region[0], crop_region[1])
    
    # Process image
    hsv = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)
    hsv = cv.GaussianBlur(hsv, (5, 5), 0)
    
    # Create color mask for yellow
    yellow_mask = cv.inRange(hsv, YELLOW[0], YELLOW[1])
    
    # Apply simple morphology to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv.morphologyEx(yellow_mask, cv.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by minimum area
    valid_contours = [c for c in contours if cv.contourArea(c) > min_contour_area]
    
    # If valid contours found, process the largest one
    if valid_contours:
        largest_contour = max(valid_contours, key=cv.contourArea)
        area = cv.contourArea(largest_contour)
        
        # Calculate center
        M = cv.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + crop_region[0][0]  # Adjust for crop offset
            
            # Calculate angle to yellow object (normalized from -1.0 to 1.0)
            image_center_x = rc.camera.get_width() / 2
            angle_to_yellow = (cx - image_center_x) / image_center_x
            angle_to_yellow = rc_utils.clamp(angle_to_yellow, -1.0, 1.0)
            
            # Only show visualization occasionally to reduce lag
            if int(time.time() * 2) % 6 == 0:  # Show every 3 seconds
                display_img = image.copy()
                adjusted_contour = largest_contour.copy()
                adjusted_contour[:, :, 1] += crop_region[0][0]
                cv.drawContours(display_img, [adjusted_contour], -1, (0, 255, 255), 2)
                cv.circle(display_img, (cx, cy), 5, (0, 0, 255), -1)
                rc.display.show_color_image(display_img)
            
            return True, (cy, cx), area
    
    # Reset angle if no yellow detected
    angle_to_yellow = 0
    return False, None, 0

def wall_following_update():
    """
    Main update function for wall following behavior
    """
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global front_distance
    global cur_state
    global speed
    global angle
    global random_number

    # Generate a random number but only occasionally
    if int(time.time()) % 5 == 0:  # Only update random number every 5 seconds
        random_number = random.randint(1, 100)
    
    # Get LIDAR samples once and reuse
    scan = rc.lidar.get_samples()
    left_angle, left_distance = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    front_angle, front_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    
    # If no point is detected in front, set front_distance to a very large value
    front_distance = 10000000 if front_dist is None else front_dist
    
    # Only print distance info occasionally to reduce console spam
    if int(time.time()) % 3 == 0:  # Every 3 seconds
        print(f"L: {left_distance:.0f}cm, R: {right_distance:.0f}cm, F: {front_distance:.0f}cm")
    
    # State machine
    if cur_state == State.move:
        move_wall_following()
    elif cur_state == State.turn:
        turn_wall_following()
    elif cur_state == State.stop:
        stop_wall_following()
    
    # Set the final speed and angle
    if left_distance > 70 and right_distance > 70 and front_distance > 190:
        speed = 1
        rc.drive.set_max_speed(1)

        # Straighten gradually
        if angle > 0:
            angle -= 0.2
        elif angle < 0:
            angle += 0.2
    elif left_distance > 70 and right_distance > 70 and front_distance < 100:
        speed = 0.8
        rc.drive.set_max_speed(0.28)
        
    # Check for yellow once per call - store result to avoid duplicate calls
    yellow_result = can_see_yellow()
    if yellow_result and yellow_result[1] is not None:
        speed = 1
        rc.drive.set_max_speed(0.35)
        if angle_to_yellow > 0:
            angle += 0.2
        elif angle_to_yellow < 0:
            angle -= 0.2
            
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    rc.drive.set_speed_angle(speed, angle)

def start():
    """
    This function is run once every time the start button is pressed
    """
    global cur_state
    global speed
    global angle
    global front_distance
    global random_number

    # Initialize front_distance to a large value
    front_distance = 10000000
    
    # Initialize car state
    speed = 0
    angle = 0
    cur_state = State.move
    
    # Print start message
    print(">> Wall Following Module Started")
    rc.drive.set_max_speed(0.28)
    
def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Update wall following
    wall_following_update()
    
    # Display current state
    states = ["MOVING", "TURNING", "STOPPED"]
    current_state_name = states[cur_state]
    
    # Draw current state on image
    image = rc.camera.get_color_image()
    if image is not None:
        h, w = image.shape[:2]
        # Draw a background rectangle
        cv.rectangle(image, (10, 10), (300, 70), (0, 0, 0), -1)
        # Put text
        cv.putText(image, f"STATE: {current_state_name}", (20, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f"Speed: {speed:.2f}, Angle: {angle:.2f}", (20, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        rc.display.show_color_image(image)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go() 