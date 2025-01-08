"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-outreach-labs

File Name: lab_d.py

Title: Lab D - Driving in Shapes

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: Create a script to enable semi-autonomous driving for the RACECAR. Button presses
enable a series of instructions sent to the RACECAR, which enable it to drive in various shapes.
Complete the lines of code under the #TODO indicators to complete the lab.

Expected Outcome: When the user runs the script, they are able to control the RACECAR
using the following keys:
- When the "A" button is pressed, drive in a circle
- When the "B" button is pressed, drive in a square
- When the "X" button is pressed, drive in a figure eight
- When the "Y" button is pressed, drive in any shape of your choice
"""

########################################################################################
# Imports
########################################################################################

import sys

from typing import Counter

sys.path.insert(1, "../../library/")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Put any global variables here

########################################################################################
# Functions
########################################################################################

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Begin at a full stop
    rc.drive.stop()

    global Counter, isDriving
    isDriving = False
    Counter = 0
    circle_started = False
  
    global isDriving1
    global isDriving2
    global isDriving3
    isDriving1 = False
    isDriving2 = False
    isDriving3 = False

    # Print start message
    # TODO (main challenge): add a line explaining what the Y button does
    print(
        ">> Lab 1 - Driving in Shapes\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Left trigger = accelerate backward\n"
        "    Left joystick = turn front wheels\n"
        "    A button = drive in a circle\n"
        "    B button = drive in a square\n"
        "    X button = drive in a figure eight\n"
    )

    global right_trigger_value, left_trigger_value, left_joystick_value
    global turn_scale, turn_angle, left_wheel_speed, right_wheel_speed
    figure_eight_started = False

def update():
    global Counter
    global isDriving
    global isDriving1
    global isDriving2
    global isDriving3

    SPEED_NORMAL = 1.0
    SPEED_SQUARE = 1
    SPEED_FIGURE_EIGHT = 1
    TURNING_NORMAL = 1.0
    TURNING_SQUARE = 1.0
    TURNING_FIGURE_EIGHT = 1

    
      
      # Check if the A button is pressed
    if rc.controller.was_pressed(rc.controller.Button.A) and not isDriving:
        isDriving = True  # Start driving in a circle
        Counter = 0  # Reset the counter

    # Check if the car should be driving in a circle
    if isDriving:
        Counter += rc.get_delta_time()

        if 0 <= Counter < 2.4:
            rc.drive.set_speed_angle(1.0, 0.0)  # Drive straight for 2.5 seconds
        elif 2.4 <= Counter < 3.75:
            rc.drive.set_speed_angle(1,0.9)
        elif 3.75 <= Counter < 4.2:
            rc.drive.set_speed_angle(1.0, 0.0)
        elif 4.2 <= Counter < 5.5:
            rc.drive.set_speed_angle(1.0, -0.9)
        elif 5.5 <= Counter < 5.9:
            rc.drive.set_speed_angle(1,0.0)
        elif 5.9 <= Counter :
            rc.drive.stop()
    # Check if the B button is pressed
    if rc.controller.was_pressed(rc.controller.Button.B) and not isDriving1:
        isDriving1 = True  # Start driving in a square
        Counter = 0  # Reset the counter

    # Check if the car should be driving in a square
    if isDriving1:
        Counter += rc.get_delta_time()

        if 0<= Counter < 3.5:
            rc.drive.set_speed_angle(1.0, 0.0)
        elif 3.5 <= Counter < 4.85:
            rc.drive.set_speed_angle(1, 0.9)
        elif 4.85 <= Counter < 8.35:
            rc.drive.set_speed_angle(1.0, 0.0)
        elif 8.35 <= Counter < 9.7:
            rc.drive.set_speed_angle(1, 1)
        elif 9.7 <= Counter <  10.2 :
            rc.drive.set_speed_angle(1.0, -0.345)
        elif 10.2 <= Counter < 11.9:
            rc.drive.set_speed_angle(1.0, 0.0)
        elif 11.9 <= Counter < 12.5:
            rc.drive.set_speed_angle(1, 0.7)
        elif 12.5 <= Counter < 12.7:
            rc.drive.set_speed_angle(1.0, 0.0)
        elif 12.7 <= Counter < 13.7:
            rc.drive.set_speed_angle(1.0, 0.8)
        elif 13.7 <= Counter < 15 :
            rc.drive.set_speed_angle(1.0, 0.0)
        elif 15 <= Counter < 16:
            rc.drive.set_speed_angle(1.0, 1)
        elif 16 <= Counter < 16.8 : 
            rc.drive.set_speed_angle(1.0, 0.0)
        elif 16.8 <= Counter :
            rc.drive.stop() # Drive straight for 3.5 seconds
              # Stop driving in a square when the duration is reached

    # Check if the X button is pressed
    if rc.controller.was_pressed(rc.controller.Button.X) and not isDriving2:
        isDriving2 = True  # Start driving in a figure-eight
        Counter = 0  # Reset the counter

    # Check if the car should be driving in a figure-eight
    if isDriving2:
        Counter += rc.get_delta_time()

        # Drive in a figure-eight
        speed = SPEED_FIGURE_EIGHT  # Adjust the speed as needed
        turning_value = TURNING_FIGURE_EIGHT  # Adjust the turning angle for clockwise motion

        # Drive straight for 2 seconds, then turn right for 1 second, then turn left for 1 second, and repeat
        if 0 <= Counter < 1.5:
            rc.drive.set_speed_angle(1,0.3)
        elif 1.5 <= Counter < 3.35:
            rc.drive.set_speed_angle(1,-0.3) 
        elif 3.3 <= Counter < 4:
            rc.drive.set_speed_angle(1,0.3)
        elif 4 <= Counter < 4.8: 
            rc.drive.set_speed_angle(1,0.8)
        elif 4.8 <= Counter < 6:
            rc.drive.set_speed_angle(1,-1)
        elif 6 <= Counter < 7.4:
            rc.drive.set_speed_angle(1,0.8)
        elif 7.4 <= Counter < 8.5:
            rc.drive.set_speed_angle(1,-0.6)
        elif 8.5 <= Counter < 9.4:
            rc.drive.set_speed_angle(1,0)
        elif 9.4 <= Counter:
            rc.drive.stop()
        
        

        

    # Check if the Y button is pressed
    if rc.controller.was_pressed(rc.controller.Button.Y) and not isDriving3:
        isDriving3 = True  # Start driving in a circle
        Counter = 0  # Reset the counter

    # Check if the car should be driving in a circle
    if isDriving3:
        Counter += rc.get_delta_time()

        if 0 <= Counter < 5.5:
            rc.drive.set_speed_angle(1,0)
        elif 5.5 <= Counter < 6.65:
            rc.drive.set_speed_angle(1,-1)
        elif 6.65 <= Counter < 9 :
            rc.drive.set_speed_angle(1,0)
        elif 9 <= Counter < 10 :
            rc.drive.set_speed_angle(1,-1)
        elif 10 <= Counter < 12.2:
            rc.drive.set_speed_angle(-1,0.3)
        elif 12.2 <= Counter < 15.6:
            rc.drive.set_speed_angle(1,-1)
        elif 15.6 <= Counter < 17 :
            rc.drive.set_speed_angle(1,0)
        elif 17 <= Counter < 18 :
            rc.drive.set_speed_angle(1,1)
        elif 18 <= Counter < 20.2 :
            rc.drive.set_speed_angle(-1,-0.3)
        elif 20.2 <= Counter < 23.6 : 
            rc.drive.set_speed_angle(1,1)
        elif 23.6 <= Counter < 24.1: 
            rc.drive.set_speed_angle(1,0)
        elif 24.1 <= Counter < 25.7:
            rc.drive.set_speed_angle(1,-1)
        elif 25.7 <= Counter < 26:
            rc.drive.set_speed_angle(1,1)
        elif 26 <= Counter < 27:
            rc.drive.set_speed_angle(1,1)
        elif 27 <= Counter < 29:
            rc.drive.set_speed_angle(-1,-0.3)
        elif 29 <= Counter < 32.5:
            rc.drive.set_speed_angle(1,1)
        elif 32.5 <= Counter < 33.2:
            rc.drive.set_speed_angle(1,0)
        elif 33.2 <= Counter < 34.5:
            rc.drive.set_speed_angle(1,-1)
        elif 34.5 <= Counter < 37:
            rc.drive.set_speed_angle(-1,0)
        elif 37 <= Counter < 40.4:
            rc.drive.set_speed_angle(1,-1)
        elif 40.4 <= Counter < 43:
            rc.drive.set_speed_angle(1,0)
        elif 43 <= Counter:
            rc.drive.stop()
            
        
        
        
        
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()


