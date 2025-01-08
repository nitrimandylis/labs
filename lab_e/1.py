"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-outreach-labs

File Name: lab_e.py

Title: Lab E - Stoplight Challenge

Author: Fuck you for making template that basicly do all the hard work << [Write your name or team name here]

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

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour (Adjust threshold!)
MIN_CONTOUR_AREA = 30

# TODO Part 1: Determine the HSV color threshold pairs for ORANGE, GREEN, RED, YELLOW, and PURPLE
# Colors, stored as a pair (hsv_min, hsv_max)
BLUE = ((90, 50, 50), (120, 255, 255))  # The HSV range for the color blue
GREEN = ((107, 81, 80), (85, 255, 255)) # The HSV range for the color green
RED = ((0, 80, 80),     (10, 255, 255))  # The HSV range for the color red
ORANGE = ((10, 100, 100), (20, 255, 255)) # The HSV range for the color orange
ANY = ((0, 1, 1), (179, 255, 255)) # The HSV range for any color
# The HSV range for the color yellow
#why yellow?
# The HSV range for the color purple  
#why purple?

# >> Variables
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0 # The area of contour
speed = 0
angle = 0

global stoplight_color
queue = [] # The queue of instructions
stoplight_color = "" # The current color of the stoplight

########################################################################################
# Functions
########################################################################################

# [FUNCTION] Finds contours in the current color image and uses them to update 
# contour_center and contour_area
def update_contour():
    
    

    print("pls no dump")

# [FUNCTION] The start function is run once every time the start button is pressed
def start():

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(0,0)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message (You may edit this to be more informative!)
    print(
        ">> Lab 3 - Stoplight Challenge\n"
        "\n"
        "Controls:\n"
        "   A button = print current speed and angle\n"
        "   B button = print contour center and area"
    )
    

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global queue
    print("update ini")

    update_contour()

    print("update fin")

    if stoplight_color == None:
        stopNow()
    elif stoplight_color == "blue":
        turnRight()
    elif stoplight_color == "orange":
        turnLeft()
    elif stoplight_color == "green":
        goStraight()
    elif stoplight_color == "red":
        stopNow()



        
    # TODO Part 2: Complete the conditional tree with the given constraints.
    
    # ... You may need more elif/else statements
    # ... FUCK U WHY MAKE THIS IN 3.9 WHY MY PRECIOUS MATCH/CASE WHYYYYYY

    # TODO Part 3: Implement a way to execute instructions from the queue once they have been placed
    # by the traffic light detector logic (Hint: Lab 2)

    # Send speed and angle commands to the RACECAR
    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)

# [FUNCTION] Appends the correct instructions to make a 90 degree right turn to the queue
def turnRight():
    global queue
    queue.append("go straight")
    

    # TODO Part 4: Complete the rest of this function with the instructions to make a right turn

# [FUNCTION] Appends the correct instructions to make a 90 degree left turn to the queue
def turnLeft():
    global queue
    queue.append("go straight")

    # TODO Part 5: Complete the rest of this function with the instructions to make a left turn

# [FUNCTION] Appends the correct instructions to go straight through the intersectionto the queue
def goStraight():
    
    global queue
    queue.append("go straight")

    # TODO Part 6: Complete the rest of this function with the instructions to make a left turn
def update_slow():    
    global queue
# [FUNCTION] Clears the queue to stop all actions
def stopNow():
    global queue
    queue.clear()

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update , update_slow)
    rc.go()