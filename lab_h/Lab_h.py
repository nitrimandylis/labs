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
    global cur_state
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

        # Visualize contours with null checks
        if contour_red is not None:
            rc_utils.draw_contour(image, contour_red)
        if contour_blue is not None:
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
            print("Pre check")
            print(f"Current State is : {cur_state}")
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
                print("Pre Check")
                print(f"Curent State : {cur_state}")
                return (cone_Selected, contour_center, contour_area, Next_Cone, contour_red)
            else:
                contour_center = rc_utils.get_contour_center(contour_blue)
                contour_area = rc_utils.get_contour_area(contour_blue)
                cone_Selected = Current_Cone.BLUE
                cur_state = State.blue 
            print("cur_state = State.blue")
            return (cone_Selected, contour_center, contour_area, Next_Cone, contour_blue)

def start():
    global current_time
    global Last_Turn
    global Distance_Cone_Red, Distance_Cone_Blue
    global angle_error
    angle_error = 0
    Last_Turn = None
    current_time = 0
    Distance_Cone_Blue = 100000000
    Distance_Cone_Red = 100000000000000
    rc.drive.set_speed_angle(1, 0)
    """Initialize the robot"""
    pass

def update():
    """Main update loop for robot control"""
    global cur_state
    global Next_Cone
    global current_Cone
    global CloseDistance
    global TrunLeftValue, TurnRightValue
    global Last_Turn    
    global current_time
    global angle_error
    global current_TurnValue
    global Time_start, Time_end
    
    # Constants for turning and distance
    Time_start = 2
    Time_end = 3
    TurnRightValue = 0.7  
    CloseDistance = 100  
    Distance_To_Start_Alinement = 160
    TrunLeftValue = -TurnRightValue
    
    
    # Get current camera images
    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()
    
    # Update contour detection
    update_contours(color_image, depth_image)
    
    # Search state - random exploration
    if cur_state == State.search:
        if contour_red is None and contour_blue is None:
            rc.drive.set_speed_angle(0.5, random.uniform(-1, 1))
        elif contour_blue is None and contour_red is not None:
            cur_state = State.red
        elif contour_red is None and contour_blue is not None:
            cur_state = State.blue
        return
    
    # Red cone handling
    if Distance_Cone_Red < Distance_Cone_Blue :
        print(f"Checking state Red: {cur_state}")
        current_time += rc.get_delta_time()
        if cur_state == State.red:
            print("Red Pass 1")
            if contour_center_red is not None:
                
                # Calculate steering angle based on cone position
                angle_error = (contour_center_red[1] - 320) / 320
                print("Distance_Red:" + str(Distance_Cone_Red))
                # If close to cone, turn right sharply
                if Distance_Cone_Red < CloseDistance and Distance_Cone_Red != 0 or Distance_Cone_Red > 10000000:
                    print("Good tone, good tone, Fox 10.")
                    current_TurnValue = TurnRightValue
                    rc.drive.set_speed_angle(0.5, current_TurnValue)
                    Last_Turn = TurnRightValue
                elif Distance_Cone_Red is not None:
                    if Distance_Cone_Red > Distance_To_Start_Alinement:
                        current_TurnValue = angle_error - 0.4
                        rc.drive.set_speed_angle(1, current_TurnValue)

                    if Distance_Cone_Red < Distance_To_Start_Alinement and Distance_Cone_Red > CloseDistance:
                        current_TurnValue = angle_error
                        rc.drive.set_speed_angle(1, current_TurnValue)
            else:
                # If red cone lost, check for blue cone or go to search
                cur_state = State.blue if contour_center_blue is not None else State.search
        
    # Blue cone handling

    if Distance_Cone_Red > Distance_Cone_Blue :
        print(f"Checking state Blue: {cur_state}")
        current_time += rc.get_delta_time()
        if cur_state == State.blue:
            print("Blue Pass 1")
            if contour_center_blue is not None:
                # Calculate steering angle based on cone position
                angle_error = (contour_center_blue[1] - 320) / 320
                print("Distance_Blue:" + str(Distance_Cone_Blue))
                # If close to cone, turn left sharply
                if Distance_Cone_Blue < CloseDistance and Distance_Cone_Blue != 0 or Distance_Cone_Blue > 10000000:
                    print("Fox 10 Blue")
                  
                    current_TurnValue = TrunLeftValue
                    rc.drive.set_speed_angle(0.5, current_TurnValue)
                    Last_Turn = TrunLeftValue
            elif Distance_Cone_Blue is not None:
                if Distance_Cone_Blue > Distance_To_Start_Alinement:
                    current_TurnValue = angle_error - 0.4
                    rc.drive.set_speed_angle(1, current_TurnValue)

                if Distance_Cone_Blue < Distance_To_Start_Alinement and Distance_Cone_Blue > CloseDistance:
                    # Otherwise, adjust to align with cone
                    current_TurnValue = angle_error
                    rc.drive.set_speed_angle(1, current_TurnValue)
        else:
            # If blue cone lost, check for red cone or go to search
            cur_state = State.red if contour_red is not None else State.search


    if Last_Turn == TurnRightValue:
        if current_time >= Time_start:
            current_TurnValue = TrunLeftValue
            rc.drive.set_speed_angle(1,current_TurnValue)
            print("Counter turn left")
            if current_time >= Time_end:
                current_time = 0.0
    elif Last_Turn == TrunLeftValue:
        if current_time >= Time_start:
            current_TurnValue = TurnRightValue
            rc.drive.set_speed_angle(1,current_TurnValue)
            print("Counter turn right")
            if current_time >= Time_end:
                current_time = 0.0  
        
    
    print(f"Current Time is :{current_time} and Distance from cones Blue:{Distance_Cone_Blue} and Red:{Distance_Cone_Red}")
    print(f"Current Turn Value is :{current_TurnValue}")
def update_slow():
    """Slow update loop for non-critical operations"""
    pass
    
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()