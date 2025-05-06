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
FRONT_WINDOW = (0,0)

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
    # Print start message
    
    print(">> Lab 4B - LIDAR Wall Following")
    rc.drive.set_max_speed(0.32)


def update():
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global cur_state
    global angle
    global speed
    global front_angle
    global front_distance

    scan = rc.lidar.get_samples()
    left_angle, left_distance = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    front_angle, front_distance = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # TODO: Follow the wall to the right of the car without hitting anything.
    print(left_distance, "<----->", right_distance)
    if cur_state == State.move:
        move()
    elif cur_state == State.turn:
        turn()
    elif cur_state == State.stop:
        stop()
    if front_distance < 60:
        # if angle > 0.4:
        #     angle += 0.1
        # elif angle < -0.3:
        #     angle -= 0.5
        print("obtc")
        print(f"front_distance: {front_distance}")
    angle = rc_utils.clamp(angle, -1, 1)
    rc.drive.set_speed_angle(speed, angle)

def move():
    global speed
    global angle
    global left_distance
    global right_distance
    global cur_state

    speed = 1
    angle = 0
    
    print("MOVE FORWARD")

    if abs(left_distance-right_distance) > 10:
        cur_state = State.turn


def turn():
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global angle
    global cur_state
    global speed

    error = right_distance - left_distance
    if error < 0:
        print("TURN LEFT")
        # angle = rc_utils.remap_range(left_angle, -60, -30, -1, 0)
        angle = rc_utils.clamp(error / 25, -1.0, 1.0)

    else:
        print("TURN RIGHT")
        # angle = rc_utils.remap_range(right_angle, 30, 60, 0, 1)

        angle = rc_utils.clamp(error / 25, -1.0, 1.0)



    if abs(error) < 10:
        cur_state == State.move
    
    speed = 0.6

def stop():
    global speed
    global angle
    global cur_state

    speed = 0
    angle = 0



########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()