import sys
import numpy as np

# Add library directory to the path
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from typing import Tuple
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

Distance_to_wall = 50  
Max_speed = 1     
Turn_speed = 1    
Window = 20   

right_wall = 60     
left_wall = 300   

def get_Distance_to_walls() -> Tuple[float, float]:
    scan = rc.lidar.get_samples()
    if scan is None:
        return 0, 0
    left_dist_Wall = rc_utils.get_lidar_average_distance(scan, left_wall, Window)
    right_dist_Wall = rc_utils.get_lidar_average_distance(scan, right_wall, Window)

    return left_dist_Wall, right_dist_Wall

def calc_angle(left_dist_Wall: float, right_dist_Wall: float) -> float:

    difference = right_dist_Wall - left_dist_Wall
    

    Norm_angle = rc_utils.remap_range(difference, -15, 15, -1, 1)

    return rc_utils.clamp(Norm_angle, -1, 1)

def start():
    """
    This function is run once every time the start button is pressed
    """
    rc.drive.stop()
def update():
    """
    Implements hybrid control for wall following
    """
    scan = rc.lidar.get_samples()
    if scan is None:
        return

    left_dist, right_dist = get_Distance_to_walls()

    angle = calc_angle(left_dist, right_dist)
    

    rc.drive.set_speed_angle(1, angle)
def update_slow():
    """
    This 
    """


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go() 