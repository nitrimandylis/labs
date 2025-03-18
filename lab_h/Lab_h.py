import sys
import cv2 as cv
import numpy as np
from enum import Enum
import random 

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()

RED = ((160, 0, 0), (179, 255, 255))
BLUE = ((90, 120, 120), (120, 255, 255))

class State(Enum):

    search = 0
    red = 1
    blue = 2
    smtng = 3

class Current_Cone(Enum):
    RED = 0
    BLUE = 1
    NOTHING = 3

cur_state = State.red
current_Cone = Current_Cone.NOTHING

speed = 0.0
angle = 0.0
last_distance = 0
counter = 0

contour_red = None
contour_blue = None
contour_center_red = None
contour_center_blue = None
contour_center = None
contour_area = 0
Distance_Cone_Red = None
Distance_Cone_Blue = None

def update_contours(image,image_depth):

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

    MIN_CONTOUR_AREA = 800

    if image is None:
        contour_center = None
        contour_area = 0
        contour_red = None
        contour_blue = None
        contour_center_red = None
        contour_center_blue = None
        Distance_Cone_Red = None
        Distance_Cone_Blue = None
        print("ERROR No image")
        return (Current_Cone.NOTHING, None, 0, None)
    else:
        contours_red = rc_utils.find_contours(image, RED[0], RED[1])
        contours_blue = rc_utils.find_contours(image, BLUE[0], BLUE[1])

        contour_red = rc_utils.get_largest_contour(contours_red, MIN_CONTOUR_AREA)
        contour_blue = rc_utils.get_largest_contour(contours_blue, MIN_CONTOUR_AREA)

        contour_center_red = None
        contour_center_blue = None
        if contour_red is not None:
            contour_center_red = rc_utils.get_contour_center(contour_red)
        if contour_blue is not None:
            contour_center_blue = rc_utils.get_contour_center(contour_blue)

        if contour_red is not None:
            rc_utils.draw_contour(image, contour_red)
        if contour_blue is not None:
            rc_utils.draw_contour(image, contour_blue)
        rc.display.show_color_image(image)

        Distance_Cone_Red = None
        Distance_Cone_Blue = None
        if image_depth is not None:
            if contour_red is not None and contour_center_red is not None:
                try:
                    Distance_Cone_Red = image_depth[contour_center_red[0]][contour_center_red[1]]
                    if Distance_Cone_Red == 0:
                        Distance_Cone_Red = 100000
                except:
                    print("Error accessing depth for red cone")
                    Distance_Cone_Red = None
            if contour_blue is not None and contour_center_blue is not None:
                try:
                    Distance_Cone_Blue = image_depth[contour_center_blue[0]][contour_center_blue[1]]
                    if Distance_Cone_Blue == 0:
                        Distance_Cone_Blue = 100000
                except:
                    print("Error accessing depth for blue cone")
                    Distance_Cone_Blue = None
                            
        if contour_red is not None and contour_blue is None:
            cur_state = State.red
            contour_center = rc_utils.get_contour_center(contour_red)
            cone_Selected = Current_Cone.RED
            print("Pre check")
            print(f"Current State is : {cur_state}")
            contour_area = rc_utils.get_contour_area(contour_red)
            return (cone_Selected, contour_center, contour_area, contour_red)

        if contour_blue is not None and contour_red is None:
            cur_state = State.blue
            contour_center = rc_utils.get_contour_center(contour_blue)
            contour_area = rc_utils.get_contour_area(contour_blue)
            cone_Selected = Current_Cone.BLUE
            print("cur_state = State.blue")
            return (cone_Selected, contour_center, contour_area, contour_blue)

        if contour_red is not None and contour_blue is not None:
            if Distance_Cone_Red is None:
                Distance_Cone_Red = 100000
            if Distance_Cone_Blue is None:
                Distance_Cone_Blue = 100000
                
            if Distance_Cone_Red < Distance_Cone_Blue:
                contour_center = rc_utils.get_contour_center(contour_red)
                contour_area = rc_utils.get_contour_area(contour_red)
                cone_Selected = Current_Cone.RED
                cur_state = State.red  
                print("Pre Check")
                print(f"Curent State : {cur_state}")
                return (cone_Selected, contour_center, contour_area, contour_red)
            else:
                contour_center = rc_utils.get_contour_center(contour_blue)
                contour_area = rc_utils.get_contour_area(contour_blue)
                cone_Selected = Current_Cone.BLUE
                cur_state = State.blue 
                print("cur_state = State.blue")
                return (cone_Selected, contour_center, contour_area, contour_blue)
        
        return (Current_Cone.NOTHING, None, 0, None)

def start():
    global current_time
    global Last_Turn
    global Distance_Cone_Red, Distance_Cone_Blue
    global angle_error
    global current_TurnValue
    global contour_red, contour_blue
    global contour_center_red, contour_center_blue
    
    angle_error = 0
    Last_Turn = None
    current_time = 0
    Distance_Cone_Red = None
    Distance_Cone_Blue = None
    current_TurnValue = None
    contour_red = None
    contour_blue = None
    contour_center_red = None
    contour_center_blue = None
    
    rc.drive.set_speed_angle(1, 0)
    pass

def update():
    global cur_state
    global current_Cone
    global CloseDistance
    global TrunLeftValue, TurnRightValue
    global Last_Turn    
    global current_time
    global angle_error
    global current_TurnValue
    global Time_start, Time_end
    global Distance_Cone_Blue, Distance_Cone_Red
    global contour_red, contour_blue
    global contour_center_red, contour_center_blue
    
    Time_start = 2
    Time_end = 2.7
    TurnRightValue = 0.7  
    CloseDistance = 70  
    Distance_To_Start_Alinement = 160
    TrunLeftValue = -TurnRightValue
    current_TurnValue = None
    
    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()
    
    update_contours(color_image, depth_image)
    
    if Distance_Cone_Red is None:
        Distance_Cone_Red = 1000000000000000
    if Distance_Cone_Blue is None:
        Distance_Cone_Blue = 100000000000000

    if cur_state == State.search:
        if contour_red is None and contour_blue is None:
            current_TurnValue = random.uniform(-1, 1)
        elif contour_blue is None and contour_red is not None:
            cur_state = State.red
        elif contour_red is None and contour_blue is not None:
            cur_state = State.blue
        if current_TurnValue is None:
            current_TurnValue = 0
        rc.drive.set_speed_angle(0.5, current_TurnValue)
        return
    
    # Red cone handling
    if Distance_Cone_Red < Distance_Cone_Blue :
        print(f"Checking state Red: {cur_state}")
        current_time += rc.get_delta_time()

        if current_time >= Time_end:
            current_time = 0
            print("Time Zeroed Red")
        if cur_state == State.red:
            print("Red Pass 1")
            if contour_center_red is not None:
                angle_error = (contour_center_red[1] - 320) / 320
                print("Distance_Red:" + str(Distance_Cone_Red))
                if Distance_Cone_Red < CloseDistance and Distance_Cone_Red != 0 or Distance_Cone_Red > 10000000:
                    print("Good tone, good tone, Fox 10.")
                    current_TurnValue = TurnRightValue
                    Last_Turn = TurnRightValue
                else:
                    if Distance_Cone_Red > Distance_To_Start_Alinement:
                        current_TurnValue = angle_error - 0.4
                    elif Distance_Cone_Red < Distance_To_Start_Alinement and Distance_Cone_Red > CloseDistance:
                        current_TurnValue = angle_error
                    else:
                        current_TurnValue = angle_error
            else:
                cur_state = State.blue if contour_center_blue is not None else State.search
        
    # Blue cone handling

    if Distance_Cone_Blue < Distance_Cone_Red :
        print(f"Checking state Blue: {cur_state}")
        current_time += rc.get_delta_time()
        if current_time >= Time_end:
            current_time = 0
            print("Time Zeroed Blue")
        if cur_state == State.blue:
            print("Blue Pass 1")
            if contour_center_blue is not None:
                angle_error = (contour_center_blue[1] - 320) / 320
                print("Distance_Blue:" + str(Distance_Cone_Blue))
                if Distance_Cone_Blue < CloseDistance and Distance_Cone_Blue != 0 or Distance_Cone_Blue > 10000000:
                    print("Fox 10 Blue")
                    current_TurnValue = TrunLeftValue
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
            cur_state = State.red if contour_red is not None else State.search


    if Last_Turn == TurnRightValue:
        if current_time >= Time_start and Distance_Cone_Blue > Distance_To_Start_Alinement and CloseDistance > Distance_Cone_Red or Distance_Cone_Red > 1000 and current_time >= Time_start and Distance_Cone_Blue > Distance_To_Start_Alinement:
            current_TurnValue = TrunLeftValue
            print("Counter turn left")
            if current_time >= Time_end:
                current_time = 0.0
    elif Last_Turn == TrunLeftValue:
        if current_time >= Time_start and Distance_Cone_Red > Distance_To_Start_Alinement and CloseDistance > Distance_Cone_Blue or Distance_Cone_Blue > 1000 and current_time >= Time_start and Distance_Cone_Red > Distance_To_Start_Alinement :
            current_TurnValue = TurnRightValue
            print("Counter turn right")
            if current_time >= Time_end:
                current_time = 0.0
    
    print(f"Current Time is :{current_time} and Distance from cones Blue:{Distance_Cone_Blue} and Red:{Distance_Cone_Red}")
    print(f"Current Turn Value is :{current_TurnValue} And Last Trun Was {Last_Turn}")
    
    if current_TurnValue is None:
        current_TurnValue = 0
    rc.drive.set_speed_angle(0.5, current_TurnValue)

def update_slow():
    pass
    
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()