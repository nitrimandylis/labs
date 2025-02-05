"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-outreach-labs

File Name: lab_e.py

Title: Lab E - Stoplight Challenge

Author: The Crackheads << [Write your name or team name here]

Purpose: Write a script to enable autonomous behavior from the RACECAR. When
the RACECAR sees a stoplight object (colored cube in the simulator), respond accordingly
by going straight, turning right, turning left, or stopping. Append instructions to the
queue depending on whether the position of the RACECAR relative to the stoplight reaches
a certain threshold, and be able to respond to traffic lights at consecutive intersections. 

Expected Outcome: When the user runs the script, the RACECAR should control itself using
the following constraints:
- When the RACECAR sees a BLUE traffic light, make a right turn at the intersection
- When the RACECAR sees an ORANGE traffic light, make a left turn at the intersection
- When the RACECAR sees a GREEN traffic light, go straight
- When the RACECAR sees a RED traffic light, stop moving,
- When the RACECAR sees any other traffic light colors, stop moving.

Considerations: Since the user is not controlling the RACECAR, be sure to consider the
following scenarios:
- What should the RACECAR do if it sees two traffic lights, one at the current intersection
and the other at the intersection behind it?
- What should be the constraint for adding the instructions to the queue? Traffic light position,
traffic light area, or both?
- How often should the instruction-adding function calls be? Once, twice, or 60 times a second?

Environment: Test your code using the level "Neo Labs > Lab 3: Stoplight Challenge".
By default, the traffic lights should direct you in a counterclockwise circle around the course.
For testing purposes, you may change the color of the traffic light by first left-clicking to 
select and then right clicking on the light to scroll through available colors.
"""
########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(1, "/Users/nick/Developer/racecar-neo-installer/racecar-student/library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

MIN_CONTOUR_AREA = 1700

BLUE = ((85, 100, 100), (105, 255, 255))
GREEN = ((40, 50, 50), (80, 255, 255))
RED = ((0, 50, 50), (10, 255, 255))
ORANGE = ((5, 50, 50), (25, 255, 255))
YELLOW = ((20, 100, 100), (30, 255, 255))
PURPLE = ((130, 50, 50), (150, 255, 255))
ANY = ((0, 0, 0), (179, 255, 255))

contour_center = None
contour_area = 0
speed = 0
angle = 0
counter = 0

global stoplight_color
queue = []
stoplight_color = ""
last_seen_color = None

Distance_Box_Blue = 0
Distance_Box_Green = 0
Distance_Box_Red = 0
Distance_Box_Orange = 0
Distance_Box_Yellow = 0
Distance_Box_Purple = 0
current_time = 0

contour_center_blue = None
contour_center_green = None
contour_center_red = None
contour_center_orange = None
contour_center_yellow = None
contour_center_purple = None

########################################################################################
# Functions
########################################################################################

def update_contour():
    global stoplight_color, contour_center, contour_area
    global contours_blue, contours_green, contours_red, contours_orange, contours_yellow, contours_purple
    global contour_blue, contour_green, contour_red, contour_orange, contour_yellow, contour_purple
    global Distance_Box_Blue, Distance_Box_Green, Distance_Box_Red, Distance_Box_Orange, Distance_Box_Yellow, Distance_Box_Purple
    global contour_center_blue, contour_center_green, contour_center_red, contour_center_orange, contour_center_yellow, contour_center_purple
    global current_time

    image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()

    if image is not None and depth_image is not None:
        contours_blue = rc_utils.find_contours(image, BLUE[0], BLUE[1])
        contours_green = rc_utils.find_contours(image, GREEN[0], GREEN[1])
        contours_red = rc_utils.find_contours(image, RED[0], RED[1])
        contours_orange = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        contours_yellow = rc_utils.find_contours(image, YELLOW[0], YELLOW[1])
        contours_purple = rc_utils.find_contours(image, PURPLE[0], PURPLE[1])

        contour_blue = rc_utils.get_largest_contour(contours_blue, MIN_CONTOUR_AREA)
        contour_green = rc_utils.get_largest_contour(contours_green, MIN_CONTOUR_AREA)
        contour_red = rc_utils.get_largest_contour(contours_red, MIN_CONTOUR_AREA)
        contour_orange = rc_utils.get_largest_contour(contours_orange, MIN_CONTOUR_AREA)
        contour_yellow = rc_utils.get_largest_contour(contours_yellow, MIN_CONTOUR_AREA)
        contour_purple = rc_utils.get_largest_contour(contours_purple, MIN_CONTOUR_AREA)

        if contour_blue is not None:
            contour_center_blue = rc_utils.get_contour_center(contour_blue)
            Distance_Box_Blue = depth_image[contour_center_blue[0]][contour_center_blue[1]]
            print("con blu")
        if contour_orange is not None:
            contour_center_orange = rc_utils.get_contour_center(contour_orange)
            Distance_Box_Orange = depth_image[contour_center_orange[0]][contour_center_orange[1]]
            print("con ora")
        if contour_green is not None:
            contour_center_green = rc_utils.get_contour_center(contour_green)
            Distance_Box_Green = depth_image[contour_center_green[0]][contour_center_green[1]]
            print("con gre")
        if contour_red is not None:
            contour_center_red = rc_utils.get_contour_center(contour_red)
            Distance_Box_Red = depth_image[contour_center_red[0]][contour_center_red[1]]
            print("con red")
        if contour_yellow is not None:
            contour_center_yellow = rc_utils.get_contour_center(contour_yellow)
            Distance_Box_Yellow = depth_image[contour_center_yellow[0]][contour_center_yellow[1]]
            print("con yel")
        if contour_purple is not None:
            contour_center_purple = rc_utils.get_contour_center(contour_purple)
            Distance_Box_Purple = depth_image[contour_center_purple[0]][contour_center_purple[1]]
            print("con pur")

        if contour_blue is not None:
            stoplight_color = "blue"
            contour_center = contour_center_blue
        elif contour_orange is not None:
            stoplight_color = "orange"
            contour_center = contour_center_orange
        elif contour_green is not None:
            stoplight_color = "green"
            contour_center = contour_center_green
        elif contour_red is not None:
            stoplight_color = "red"
            contour_center = contour_center_red
        elif contour_yellow is not None:
            stoplight_color = "yellow"
            contour_center = contour_center_yellow
        elif contour_purple is not None:
            stoplight_color = "purple"
            contour_center = contour_center_purple
        else:
            stoplight_color = None
    else:
        stoplight_color = None


    current_time += rc.get_delta_time()
    if int(current_time) % 2 == 0:
        rc.display.show_color_image(image)
        print("clorImage"+ f"current time: {current_time}")
    if int(current_time) % 2 == 1:
        rc.display.show_depth_image(depth_image)
        print("depthImage" + f"current time: {current_time}")

def start():
    global counter
    counter = 0

    rc.drive.set_speed_angle(0,0)
    rc.set_update_slow_time(0.5)

    print(
        ">> Lab 3 - Stoplight Challenge\n"
        "\n"
        "Controls:\n"
        "   A button = print current speed and angle\n"
        "   B button = print contour center and area"
    )

def update():
    global queue, counter, last_seen_color
    update_contour()
    print(f"stoplight color: {stoplight_color}")

    if stoplight_color in ["blue", "orange", "green", "red", "yellow", "purple"]:
        last_seen_color = stoplight_color
        action = {"blue": turnRight, "orange": turnLeft, "green": goStraight, "red": stopNow, "yellow": stopNow, "purple": stopNow}[stoplight_color]
        action()
        print(f"{action.__name__}")
    elif stoplight_color is None:
        if last_seen_color in ["blue", "orange", "green", "red", "yellow", "purple"]:
            action = {"blue": turnRight, "orange": turnLeft, "green": goStraight, "red": stopNow, "yellow": stopNow, "purple": stopNow}[last_seen_color]
            action()
            print(f"{action.__name__} ls")
        else:
            rc.drive.set_speed_angle(1.0, 0)

    counter += rc.get_delta_time()
    print(f"last seen color: {last_seen_color}")

    print(f"Distance Box Blue: {Distance_Box_Blue}")
    print(f"Distance Box Green: {Distance_Box_Green}")
    print(f"Distance Box Red: {Distance_Box_Red}")
    print(f"Distance Box Orange: {Distance_Box_Orange}")
    print(f"Distance Box Yellow: {Distance_Box_Yellow}")
    print(f"Distance Box Purple: {Distance_Box_Purple}")
def turnRight():
   global queue
   global counter
   if Distance_Box_Blue > 70:
    Angle_error = (contour_center_blue[1] - 320) / 320
    angle = 0.5 * Angle_error
    rc.drive.set_speed_angle(1,angle)
   if Distance_Box_Blue < 70 or Distance_Box_Blue == 0:
    counter += rc.get_delta_time()
    
    if 0 < counter < 1.5:
        rc.drive.set_speed_angle(1,1)
    elif 1.5 < counter < 2.3:
        rc.drive.set_speed_angle(1,1)
    elif 2.3 < counter < 3:
        rc.drive.set_speed_angle(1,0)
    elif counter > 3:
        rc.drive.set_speed_angle(1,0)
        counter = 0
        print("counter reset right")

   # TODO Part 4: Complete the rest of this function with the instructions to make a right turn


# [FUNCTION] Appends the correct instructions to make a 90 degree left turn to the queue
def turnLeft():
   global queue
   global counter
   if Distance_Box_Orange > 70:
    Angle_error = (contour_center_orange[1] - 320) / 320
    angle = 0.5 * Angle_error
    rc.drive.set_speed_angle(1,angle)
   if Distance_Box_Orange < 70 or Distance_Box_Orange == 0:
    counter += rc.get_delta_time() 
     # Increment the counter
    if 0 <= counter < 1.5:
        rc.drive.set_speed_angle(1, -1)  # Turn left
    elif 1.5 <= counter < 2.3:
        rc.drive.set_speed_angle(1, -1)  # Turn left
    elif 2.3 <= counter < 3:
        rc.drive.set_speed_angle(1, 0)  # Move forward
    elif counter >= 3:
       rc.drive.set_speed_angle(1, 0)
       counter = 0 
       print("counter reset left") # Stop the car
   # Append the instruction to the queue

# [FUNCTION] Appends the correct instructions to go straight through the intersectionto the queue
def goStraight():
   global queue
   Angle_error = (contour_center_green[1] - 320) / 320
   angle = 0.5 * Angle_error
   rc.drive.set_speed_angle(1,angle)
 

def stopNow():
   global queue
   Angle_error = (contour_center_red[1] - 320) / 320
   angle = 0.5 * Angle_error
   rc.drive.set_speed_angle(0,angle)


   # TODO Part 6: Complete the rest of this function with the instructions to make a left turn
def update_slow():    
   global queue
# [FUNCTION] Clears the queue to stop all actions


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
   rc.set_start_update(start, update , update_slow)
   rc.go()
