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
    """
    An enumeration of the possible states of the racecar.
    """
    search = 0
    red = 1
    blue = 2
    smtng = 3


class Next_Cone(Enum):
    RED = 0
    BLUE = 1
    NOTHING = 3

class Current_Cone(Enum):
    RED = 0
    BLUE = 1
    NOTHING = 3


cur_state = State.search
Next_Cone = Next_Cone.NOTHING
current_Cone = Current_Cone.NOTHING


speed = 0.0
angle = 0.0
last_distance = 0
counter = 0


def update_contours(image,imag_depth):

    MIN_CONTOUR_AREA = 800

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the contours
        contours_red = rc_utils.find_contours(image, RED[0], RED[1])

        contours_blue = rc_utils.find_contours(image, BLUE[0], BLUE[1])

        contour_red = rc_utils.get_largest_contour(contours_red, MIN_CONTOUR_AREA)

        contour_blue = rc_utils.get_largest_contour(contours_blue, MIN_CONTOUR_AREA)

        contour_center_red = rc_utils.get_contour_center(contour_red)
        contour_center_blue = rc_utils.get_contour_center(contour_blue)

        if contour_red is not None and contour_blue is None:
            cur_state = State.red
            Next_Cone = Next_Cone.BLUE
            contour_center = rc_utils.get_contour_center(contour_red)
            cone_Selected = Current_Cone.RED
            contour_area = rc_utils.get_contour_area(contour_red)

            return (cone_Selected, contour_center, contour_area, Next_Cone, contour_red)


        if contour_blue is not None and contour_red is None:
            cur_state = State.blue
            Next_Cone = Next_Cone.RED
            contour_center = rc_utils.get_contour_center(contour_blue)
            contour_area = rc_utils.get_contour_area(contour_blue)
            cone_Selected = Current_Cone.BLUE

            return (cone_Selected, contour_center, contour_area, Next_Cone, contour_blue)

        if image_depth is not None:

            if contour_red is not None:
                Distance_Cone_Red = image_depth[contour_center_red[0]][contour_center_red[1]]
            if contour_blue is not None:
                Distance_Cone_Blue = image_depth[contour_center_blue[0]][contour_center_blue[1]]
            


        if contour_red is not None and contour_blue is not None:
            if Distance_Cone_Red < Distance_Cone_Blue:
                contour_center = rc_utils.get_contour_center(contour_red)
                contour_area = rc_utils.get_contour_area(contour_red)
                cone_Selected = Current_Cone.RED
                cur_state = State.red
                return (cone_Selected, contour_center, contour_area, Next_Cone, contour_red)
            else:
                contour_center = rc_utils.get_contour_center(contour_blue)
                contour_area = rc_utils.get_contour_area(contour_blue)
                cone_Selected = Current_Cone.BLUE
                cur_state = State.blue
                return (cone_Selected, contour_center, contour_area, Next_Cone, contour_blue)
    
    current_time += rc.get_delta_time()
    if int(current_time) % 2 == 0:
        rc.display.show_color_image(image)
        print("clorImage"+ f"current time: {current_time}")
    if int(current_time) % 2 == 1:
        rc.display.show_depth_image(depth_image)
        print("depthImage" + f"current time: {current_time}")

def start():
    pass

def update():
    
    global cur_state
    global Next_Cone
    global current_Cone
    global CloseDistance

    CloseDistance = 50

    update_contours(rc.camera.get_color_image(), rc.camera.get_depth_image())

    if cur_state == State.search:
        Random_angle = random.uniform(-1,1)
   #     Random_speed = random.uniform(-1,1)
        rc.drive.set_speed_angle(1, Random_angle)

    if cur_state == State.red:
        Angle_error_blue = (contour_center_blue[1] - 320) / 320
        angle = 0.5 * Angle_error_blue
        rc.drive.set_speed_angle(1,angle)
        if Distance_Cone_Red < CloseDistance:
            rc.drive.set_speed_angle(1,1)

        

        


def update_slow():
    pass


if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()