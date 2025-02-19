# Import required libraries
import sys
import cv2 as cv
import numpy as np
from enum import Enum
import random 

# Add library path and import racecar modules
sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

# Initialize racecar
rc = racecar_core.create_racecar()

# Define color ranges for cone detection
RED = ((160, 0, 0), (179, 255, 255))  # HSV range for red color
BLUE = ((90, 120, 120), (120, 255, 255))  # HSV range for blue color

class State(Enum):
    """
    An enumeration of the possible states of the racecar.
    """
    search = 0  # State for searching cones
    red = 1     # State when following red cone
    blue = 2    # State when following blue cone
    smtng = 3   # Undefined state

# Enums for tracking next and current cone colors
class Next_Cone(Enum):
    RED = 0
    BLUE = 1
    NOTHING = 3

class Current_Cone(Enum):
    RED = 0
    BLUE = 1
    NOTHING = 3

# Initialize state variables
cur_state = State.red
Next_Cone = Next_Cone.NOTHING
current_Cone = Current_Cone.NOTHING

# Initialize control variables
speed = 0.0
angle = 0.0
last_distance = 0
counter = 0

def update_contours(image,image_depth):
    """
    Updates cone detection and tracking using image processing
    Args:
        image: Color image from camera
        image_depth: Depth image from camera
    Returns:
        Tuple containing cone information (selected cone, center, area, next cone, contour)
    """
    global Next_Cone
    global current_Cone
    global Distance_Cone_Red
    global Distance_Cone_Blue
    global contour_center
    global contour_area
    global contour_red
    global contour_blue
    global contour_center_red
    global contour_center_blue
    global MIN_CONTOUR_AREA

    MIN_CONTOUR_AREA = 800  # Minimum area threshold for valid contours

    if image is None:
        contour_center = None
        contour_area = 0
        print("ERROR No image")
    else:
        # Process image to find red and blue contours
        contours_red = rc_utils.find_contours(image, RED[0], RED[1])
        contours_blue = rc_utils.find_contours(image, BLUE[0], BLUE[1])

        # Get largest contours for each color
        contour_red = rc_utils.get_largest_contour(contours_red, MIN_CONTOUR_AREA)
        contour_blue = rc_utils.get_largest_contour(contours_blue, MIN_CONTOUR_AREA)

        # Calculate centers of contours
        contour_center_red = rc_utils.get_contour_center(contour_red)
        contour_center_blue = rc_utils.get_contour_center(contour_blue)

        # Visualize contours
        rc_utils.draw_contour(image, contour_red)
        rc_utils.draw_contour(image, contour_blue)
        rc.display.show_color_image(image)

        # Process depth information if available
        if image_depth is not None:
            if contour_red is not None:
                Distance_Cone_Red = image_depth[contour_center_red[0]][contour_center_red[1]]
            if contour_blue is not None:
                Distance_Cone_Blue = image_depth[contour_center_blue[0]][contour_center_blue[1]]
                            
        # Handle case: only red cone visible
        if contour_red is not None and contour_blue is None:
            cur_state = State.red
            Next_Cone = Next_Cone.BLUE
            contour_center = rc_utils.get_contour_center(contour_red)
            cone_Selected = Current_Cone.RED
            print("cur_state = State.red")
            contour_area = rc_utils.get_contour_area(contour_red)
            return (cone_Selected, contour_center, contour_area, Next_Cone, contour_red)

        # Handle case: only blue cone visible
        if contour_blue is not None and contour_red is None:
            cur_state = State.blue
            Next_Cone = Next_Cone.RED
            contour_center = rc_utils.get_contour_center(contour_blue)
            contour_area = rc_utils.get_contour_area(contour_blue)
            cone_Selected = Current_Cone.BLUE
            print("cur_state = State.blue")
            return (cone_Selected, contour_center, contour_area, Next_Cone, contour_blue)

        # Handle case: both cones visible - choose closest
        if contour_red is not None and contour_blue is not None:
            if Distance_Cone_Red < Distance_Cone_Blue:
                contour_center = rc_utils.get_contour_center(contour_red)
                contour_area = rc_utils.get_contour_area(contour_red)
                cone_Selected = Current_Cone.RED
                cur_state = State.red  
                print("cur_state = State.red")
                return (cone_Selected, contour_center, contour_area, Next_Cone, contour_red)
            else:
                contour_center = rc_utils.get_contour_center(contour_blue)
                contour_area = rc_utils.get_contour_area(contour_blue)
                cone_Selected = Current_Cone.BLUE
                cur_state = State.blue 
            print("cur_state = State.blue")
            return (cone_Selected, contour_center, contour_area, Next_Cone, contour_blue)

def start():
    """Initialize the robot"""
    pass

def update():
    """Main update loop for robot control"""
    global cur_state
    global Next_Cone
    global current_Cone
    global CloseDistance
    global TrunLeftValue, TurnRightValue
    
    # Constants for turning and distance
    TurnRightValue = 0.7  # Increased turning angle for sharper turns
    CloseDistance = 100   # Increased detection distance for earlier turning
    TrunLeftValue = -TurnRightValue
    
    # Get current camera images
    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()
    
    # Update contour detection
    update_contours(color_image, depth_image)
    
    # Search state - random exploration
    if cur_state == State.search:
        rc.drive.set_speed_angle(0.5, random.uniform(-1, 1))
        return
    
    # Red cone handling
    if cur_state == State.red:
        print("this worijfjk")
        if contour_center_red is not None:
            
            # Calculate steering angle based on cone position
            angle_error = (contour_center_red[1] - 320) / 320
            print("Distance_Red:" + str(Distance_Cone_Red))
            # If close to cone, turn right sharply
            if Distance_Cone_Red < CloseDistance:
                rc.drive.set_speed_angle(0.5, TurnRightValue)
            else:
                # Otherwise, adjust to align with cone
                rc.drive.set_speed_angle(1, 0.5 * angle_error)
        else:
            # If red cone lost, check for blue cone or go to search
            cur_state = State.blue if contour_center_blue is not None else State.search
    
    # Blue cone handling
    if cur_state == State.blue:
        if contour_center_blue is not None:
            # Calculate steering angle based on cone position
            angle_error = (contour_center_blue[1] - 320) / 320
            print("Distance_Blue:" + str(Distance_Cone_Blue))
            # If close to cone, turn left sharply
            if Distance_Cone_Blue < CloseDistance:
                rc.drive.set_speed_angle(0.5, TrunLeftValue)
            else:
                # Otherwise, adjust to align with cone
                rc.drive.set_speed_angle(1, 0.5 * angle_error)
        else:
            # If blue cone lost, check for red cone or go to search
            cur_state = State.red if contour_center_red is not None else State.search
    
    

def update_slow():
    """Slow update loop for non-critical operations"""
    pass
    
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()