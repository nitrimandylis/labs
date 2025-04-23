"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020
Lab 4B - LIDAR Wall Following
"""

########################################################################################
# Imports
########################################################################################

from random import seed
import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here

LEFT_WINDOW = (-45, -15)
RIGHT_WINDOW = (15, 45)
FRONT_WINDOW = (-10, 10)  # Define a window for directly in front of the car

class State(IntEnum):
    move = 0
    turn = 1
    stop = 2

cur_state = State.move

speed = 0.0
angle = 0.0
left_angle = 0
left_distance = 0
right_angle = 0
right_distance = 0
front_distance = 10000000  # Initialize to a very large value

########################################################################################
# Functions
########################################################################################

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()
    global cur_state
    global speed
    global angle
    global front_distance
    
    # Initialize front_distance to a large value
    front_distance = 10000000
    
    # Print start message
    print(">> Lab 4B - LIDAR Wall Following")
    rc.drive.set_max_speed(0.32)


def update():
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global front_distance
    global cur_state
    global speed
    global angle

    scan = rc.lidar.get_samples()
    left_angle, left_distance = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    
    # Get the distance to the closest point directly in front of the car
    front_angle, front_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    
    # If no point is detected in front, set front_distance to a very large value
    if front_dist is None:
        front_distance = 10000000
    else:
        front_distance = front_dist
    
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Print distance information
    print(f"Left: {left_distance:.1f}cm, Right: {right_distance:.1f}cm, Front: {front_distance:.1f}cm")
    
    # State machine
    if cur_state == State.move:
        move()
    elif cur_state == State.turn:
        turn()
    elif cur_state == State.stop:
        stop()
    
    # Set the final speed and angle
    if left_distance > 70 and right_distance > 70 and front_distance > 160:
        speed = 1
        rc.drive.set_max_speed(1)
        if angle > 0:
            angle -= 0.4
        elif angle < 0:
            angle += 0.4
    else:
        speed = 0.7
        rc.drive.set_max_speed(0.32)
    rc.drive.set_speed_angle(speed, angle)

def move():
    global speed
    global angle
    global left_distance
    global right_distance
    global front_distance
    global cur_state

    speed = 1
    angle = 0
    
    print("MOVE FORWARD")

    # Check if we need to turn based on wall distances
    if abs(left_distance-right_distance) > 10:
        cur_state = State.turn
    
    # Check if we need to stop or slow down based on front distance



def turn():
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
        print("TURN LEFT")
        angle = rc_utils.clamp(error / 25, -1.0, 1.0)
        if angle > 0:
            angle += 0.4
        else:
            angle -= 0.4
        angle = rc_utils.clamp(angle, -1.0, 1.0)
    else:
        print("TURN RIGHT")
        angle = rc_utils.clamp(error / 25, -1.0, 1.0)
        if angle > 0:
            angle += 0.4
        else:
            angle -= 0.4
        angle = rc_utils.clamp(angle, -1.0, 1.0)

    # If front distance is small, prioritize avoiding the obstacle
    if front_distance < 30:
        if left_distance > right_distance:
            angle = -1.0  # Turn sharp left
        else:
            angle = 1.0   # Turn sharp right
        print("Obstacle ahead! Emergency turn!")

    if abs(error) < 10:
        cur_state = State.move
    
    speed = 0.7

def stop():
    global speed
    global angle
    global cur_state
    global front_distance

    speed = 0
    angle = 0
    
    # If the path is clear again, start moving
    if front_distance > 40:
        cur_state = State.move
        print("Path is clear, resuming movement")

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()