"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 6A - Grand Prix Driver
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import math
import random

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

class State(IntEnum):
    LANE_FOLLOWING = 0
    LANE_FOLLOWING_ID_0 = 1
    LANE_FOLLOWING_ID_1 = 2
    WALL_FOLLOWING_ID_3 = 3

speed = 0
angle = 0
contour_center = None
contour_area = 0
wall_following = False
RED = ((170, 50, 50), (10, 255, 255))
BLUE = ((100, 150, 50), (110, 255, 255))
GREEN = ((40, 60, 60), (80, 255, 255))
ORANGE = ((10, 50, 50), (20, 255, 255))
PURPLE = ((110, 59, 50), (165, 255, 255))
ID = 0

# Variables for marker detection and tracking
previous_ID = 0
marker_timeout = 0
turning_timer = 0
current_time = 0  # Track current time since start
is_turning_right = False
distance_to_marker = 10000  # Initialize to a very large value
contour_corners = None      # Store marker corners
COLOR = "none"              # Store marker color
Slow_oreint = "NONE"        # Initialize orientation for slow turns

########################################################################################
# Functions
########################################################################################

isSimulation = True
import math
import copy
import cv2 as cv
import numpy as np
from typing import Any, Tuple, List, Optional
from enum import Enum
from enum import IntEnum

# Import Racecar library
sys.path.append("../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar(True)

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
front_distance = 1000

potential_colors = [
    ((10, 50, 50), (20, 255, 255),'ORANGE'),
    ((100, 150, 50), (110, 255, 255),'BLUE'),
    ((40, 50, 50), (80, 255, 255),'GREEN'),  # The HSV range for the color green
    ((170, 50, 50), (10, 255, 255),'RED'),
    ((110, 59, 50), (165, 255, 255),'PURPLE')
]

class Orientation(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

class ARMarker:
    
    def __init__(self, marker_id, marker_corners):
        self.__id = marker_id
        self.__corners = marker_corners
              
        if self.__corners[0][1] > self.__corners[2][1]:
            if self.__corners[0][0] > self.__corners[2][0]:
                self.__orientation = Orientation.DOWN
            else:
                self.__orientation = Orientation.RIGHT
        else:
            if self.__corners[0][0] > self.__corners[2][0]:
                self.__orientation = Orientation.LEFT
            else:
                self.__orientation = Orientation.UP
                
        # Create fields to store the detected color and the area of that color's contour
        self.__color = "not detected"
        self.__color_area = 0
        
    def detect_colors(self, image, potential_colors):
        if image is None:
            return
            
        marker_top, marker_left = self.__corners[self.__orientation.value]
        marker_bottom, marker_right = self.__corners[(self.__orientation.value + 2) % 4]
        half_marker_height = (marker_bottom - marker_top) // 2
        half_marker_width = (marker_right - marker_left) // 2
        crop_top_left = (
            max(0, marker_top - half_marker_height),
            max(0, marker_left - half_marker_width),
        )
        crop_bottom_right = (
            min(image.shape[0], marker_bottom + half_marker_height) + 1,
            min(image.shape[1], marker_right + half_marker_width) + 1,
        )
        cropped_image = rc_utils.crop(image, crop_top_left, crop_bottom_right)
        
        for (hsv_lower, hsv_upper, color_name) in potential_colors:
            contours = rc_utils.find_contours(cropped_image, hsv_lower, hsv_upper) 
            largest_contour = rc_utils.get_largest_contour(contours)
            if largest_contour is not None:
                contour_area = rc_utils.get_contour_area(largest_contour)
                if contour_area > self.__color_area:
                    self.__color_area = contour_area
                    self.__color = color_name
            
    def get_id(self):
        return self.__id
    
    def get_corners(self):
        return self.__corners
    
    def get_orientation(self):
        return self.__orientation
    
    def get_color(self):
        return self.__color
    
    def __str__(self):
        return f"ID: {self.__id}\nCorners: {self.__corners}\nOrientation: {self.__orientation}\nColor: {self.__color}"

def get_ar_markers(image):
    if image is None:
        return []
        
    # Handle both older and newer versions of OpenCV
    try:
        # Try using newer OpenCV ArUco API
        if hasattr(cv, 'aruco'):
            # Check if we're using OpenCV 4.7.0+ with updated ArUco API
            if hasattr(cv.aruco, 'Dictionary_get'):
                dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
                parameters = cv.aruco.DetectorParameters_create()
                aruco_data = cv.aruco.detectMarkers(image, dictionary, parameters=parameters)
            else:
                # For newer OpenCV versions (4.7.0+)
                dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
                detector = cv.aruco.ArucoDetector(dictionary)
                corners, ids, _ = detector.detectMarkers(image)
                aruco_data = (corners, ids)
        else:
            print("ArUco module not found in OpenCV installation")
            return []
        
        # A list of ARMarker objects representing the AR markers found in aruco_data
        markers = []
        
        # Check if markers were detected (handle both API versions)
        if (isinstance(aruco_data, tuple) and len(aruco_data) >= 2 and len(aruco_data[0]) > 0 and aruco_data[1] is not None) or \
           (isinstance(corners, list) and len(corners) > 0 and ids is not None):
            
            # Get the actual data based on which API was used
            if 'corners' in locals() and 'ids' in locals():
                # New API
                marker_corners = corners
                marker_ids = ids
            else:
                # Old API
                marker_corners = aruco_data[0]
                marker_ids = aruco_data[1]
            
            for i in range(len(marker_corners)):
                corners = marker_corners[i][0].astype(np.int32)
                for j in range(len(corners)):
                    col = corners[j][0]
                    corners[j][0] = corners[j][1]
                    corners[j][1] = col
                marker_id = marker_ids[i][0]
                
                markers.append(ARMarker(marker_id, corners))
                markers[-1].detect_colors(image, potential_colors)
            
        return markers
    except Exception as e:
        print(f"Error detecting AR markers: {e}")
        import traceback
        traceback.print_exc()
        return []

def ar_info(marker: ARMarker):
    if marker.get_color() == 'PURPLE' or marker.get_color() == 'ORANGE':
        return f'{marker.get_color()} Lane Following'
    if marker.get_id() == 0:
        return 'Turn Left'
    if marker.get_id() == 1:
        return 'Turn Right'
    if marker.get_id() == 199:
        if marker.get_orientation() == Orientation.LEFT:
            return 'Turn Left'
        if marker.get_orientation() == Orientation.RIGHT:
            return 'Turn Right'
    if marker.get_id() == 2:
        if marker.get_color() == 'not detected':
            return 'Slalom'
    if marker.get_id() == 3:
        return 'wall following'
    return f'Follow {marker.get_color()} line'

def get_markers_info():
    image = rc.camera.get_color_image()
    if image is None:
        return []

    markers = get_ar_markers(image)
    msgs = []
    for i in markers:
        msgs.append(ar_info(i))
    return msgs

def highlight_markers(image, markers):
    """Add visual highlighting to markers in the image"""
    if image is None or not markers:
        return image
    
    # Create a copy of the image to draw on
    marked_image = image.copy()
    
    for marker in markers:
        corners = marker.get_corners()
        
        # Draw a rectangle around the marker
        cv.line(marked_image, (corners[0][1], corners[0][0]), (corners[1][1], corners[1][0]), (0, 255, 0), 2)
        cv.line(marked_image, (corners[1][1], corners[1][0]), (corners[2][1], corners[2][0]), (0, 255, 0), 2)
        cv.line(marked_image, (corners[2][1], corners[2][0]), (corners[3][1], corners[3][0]), (0, 255, 0), 2)
        cv.line(marked_image, (corners[3][1], corners[3][0]), (corners[0][1], corners[0][0]), (0, 255, 0), 2)
        
        # Calculate center of marker
        center_x = sum(corner[1] for corner in corners) // 4
        center_y = sum(corner[0] for corner in corners) // 4
        
        # Display marker info
        marker_info = f"ID: {marker.get_id()}, Color: {marker.get_color()}"
        instruction = ar_info(marker)
        
        # Add text with a background for readability
        cv.putText(marked_image, marker_info, (center_x - 10, center_y - 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv.putText(marked_image, marker_info, (center_x - 10, center_y - 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv.putText(marked_image, instruction, (center_x - 10, center_y + 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv.putText(marked_image, instruction, (center_x - 10, center_y + 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return marked_image

def print_info():
    """Print information about the marker detector"""
    print("AR Marker Detector Started")
    print("=========================")
    print("Running in continuous detection mode")
    
    # Print OpenCV version for debugging
    print(f"OpenCV version: {cv.__version__}")

def process_detection():
    """Process marker detection in continuous mode"""
    current_image = rc.camera.get_color_image()
    if current_image is not None:
        markers = get_ar_markers(current_image)
        if markers:
            # Highlight markers on image
            current_image = highlight_markers(current_image, markers)
        return current_image
    return None

def print_current_markers():
    """Print information about currently detected markers"""
    try:
        image = rc.camera.get_color_image()
        if image is not None:
            markers = get_ar_markers(image)
            if markers:
                print(f"Currently seeing {len(markers)} markers")
                for marker in markers:
                    print(f"  - ID: {marker.get_id()}, Color: {marker.get_color()} , Orientation: {marker.get_orientation()}")
    except Exception as e:
        print(f"Error in marker detection: {e}")

def calculate_marker_distance(marker_corners):
    """
    Calculate the approximate distance to a marker based on its apparent size
    
    Args:
        marker_corners: The corners of the marker
    
    Returns:
        float: Estimated distance to marker in cm
    """
    if marker_corners is None:
        return 10000  # Return a large value if no corners
    
    # Use area of the rectangle as a proxy for distance
    x_coords = [corner[1] for corner in marker_corners]
    y_coords = [corner[0] for corner in marker_corners]
    
    # Calculate width and height
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    # Calculate area
    marker_area = width * height
    
    # Return distance estimate based on area
    # The constants can be calibrated for better accuracy
    if marker_area <= 0:
        return 10000
    
    # Marker size is inversely proportional to distance
    # 100000 is a scaling factor that can be adjusted
    estimated_distance = 100000 / marker_area
    
    return estimated_distance

def save_current_markers():
    global ID, COLOR, distance_to_marker, contour_corners , Orientation
    
    image = rc.camera.get_color_image()
    if image is None:
        return False
        
    markers = get_ar_markers(image)
    if not markers:
        return False
        
    for marker in markers:
        ID = marker.get_id()
        COLOR = marker.get_color()
        Orientation = marker.get_orientation()
        contour_corners = marker.get_corners()
        
        # Calculate distance to this marker
        distance_to_marker = calculate_marker_distance(contour_corners)
        
        print(f"ID: {ID}, COLOR: {COLOR}, Distance: {distance_to_marker:.1f}cm")
        return True  # Mark that we found a marker
        
    return False  # No marker found

def start():
    """Initialize the marker detector"""
    global current_time, counter, Slow_oreint
    print_info()
    
    # Initialize current_time
    current_time = 0
    counter = 0
    Slow_oreint = "NONE"
    
    # Set update rate for slow update
    rc.set_update_slow_time(10)

def update():
    """Main update function for marker detection"""
    global Slow_oreint, ID, previous_ID, marker_timeout, turning_timer, is_turning_right, contour_corners, distance_to_marker, current_time , counter
    
    # Update current time
    current_time += rc.get_delta_time()
    
    # Process and display markers
    current_image = process_detection()
    if current_image is not None:
        rc.display.show_color_image(current_image)

    # Save current markers and get marker info
    save_current_markers()
    print_current_markers()
    marker = get_ar_markers(rc.camera.get_color_image())

    # Only process the marker logic if we have valid marker data
    
    if ID == 3:
        WALL_FOLLOWING_UPDATE_ID_3()
    print("distance_to_marker: ", distance_to_marker)
    if ID == 199 and previous_ID == 3 and Orientation == Orientation.RIGHT:
        # Slow_oreint = "RIGHT"# Now we can directly use distance_to_marker which is already calculated
        WALL_FOLLOWING_UPDATE_ID_3()
        

        if distance_to_marker < 30:
            # Start turning right
            angle = 1.0  # Full right turn
            rc.drive.set_speed_angle(1, angle)
            # If we weren't turning before, start the timer
            counter += rc.get_delta_time()
            if counter > 0.3:
                if previous_ID == 3:
                    WALL_FOLLOWING_UPDATE_ID_3()
                    previous_ID = 3
                    ID = previous_ID
                    counter = 0
            
            print(f"Marker 199 detected - turning right! Distance: {distance_to_marker:.1f}cm")
    if ID == 199 and previous_ID == 3 and Orientation == Orientation.LEFT:
        # Slow_oreint = "LEFT"# Now we can directly use distance_to_marker which is already calculated
        WALL_FOLLOWING_UPDATE_ID_3()
        
        

        if distance_to_marker < 30:
            # Start turning right
            angle = -1.0  # Full right turn
            rc.drive.set_speed_angle(1, angle)
            # If we weren't turning before, start the timer
            counter += rc.get_delta_time()
            if counter > 0.3:
                if previous_ID == 3:
                    WALL_FOLLOWING_UPDATE_ID_3()
                    
                    previous_ID = 3
                    ID = previous_ID
                    counter = 0
            
            print(f"Marker 199 detected - turning left! Distance: {distance_to_marker:.1f}cm")
        # Remember this marker ID
        
        

def stop_WALL_FOLLOWING_ID_3():
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

def turn_WALL_FOLLOWING_ID_3():
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global front_distance
    global angle
    global cur_state
    global speed
    global counter

    error = right_distance - left_distance
    if error < 0:
        print("TURN LEFT")
        angle = rc_utils.clamp(error / 25, -1.0, 1.0)
        if angle > 0 and angle < 0.4:
            angle += 0.1
        elif angle < 0 and angle > -0.4:
            angle -= 0.1
        if angle > 0.4:
            angle += 0.4
        elif angle < -0.4:
            angle -= 0.4
        angle = rc_utils.clamp(angle, -1.0, 1.0)
        rc.drive.set_max_speed(0.32)
    else:   
        print("TURN RIGHT")
        angle = rc_utils.clamp(error / 25, -1.0, 1.0)
        if angle > 0 and angle < 0.4:
            angle += 0.1
        elif angle < 0 and angle > -0.4:
            angle -= 0.1
        if angle > 0.4:
            angle += 0.4
        elif angle < -0.4:
            angle -= 0.4
        angle = rc_utils.clamp(angle, -1.0, 1.0)
        rc.drive.set_max_speed(0.32)
    # If front distance is small, prioritize avoiding the obstacle
    # if front_distance < 30:
    #     if left_distance > right_distance:
    #         angle = -1.0  # Turn sharp left
    #     else:
    #         angle = 1.0   # Turn sharp right
    #     print("Obstacle ahead! Emergency turn!")
    if front_distance < 30 and left_distance == right_distance:
        print("OBSTACLE AHEAD")
        angle = 1

    if abs(error) < 10:
        cur_state = State.move
    
    speed = 0.7

def move_WALL_FOLLOWING_ID_3():
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

def WALL_FOLLOWING_UPDATE_ID_3():


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
        move_WALL_FOLLOWING_ID_3()
    elif cur_state == State.turn:
        turn_WALL_FOLLOWING_ID_3()
    elif cur_state == State.stop:
        stop_WALL_FOLLOWING_ID_3()
    
    # Set the final speed and angle
    if left_distance > 70 and right_distance > 70 and front_distance > 190:
        speed = 1
        print("max POWWWWWWWARRRR")
        rc.drive.set_max_speed(1)

        if angle > 0:
            angle -= 0.2
        elif angle < 0:
            angle += 0.2
    elif left_distance > 70 and right_distance > 70 and front_distance < 100:
        print("Lower POWWWWWWWARRRR")
        speed = 0.7
        rc.drive.set_max_speed(0.3)
    
    # Get the image outside the conditional to avoid UnboundLocalError
    # image = rc.camera.get_color_image()
    # if image is not None:
    #     YELLOW = ((0, 100, 100), (30, 255, 255))
    #     contours_yellow = rc_utils.find_contours(image, YELLOW[0], YELLOW[1])
    #     if contours_yellow is not None:
    #         wall_Finish_yellow = rc_utils.get_largest_contour(contours_yellow)
    #         if wall_Finish_yellow is not None:
    #             print("YELLOW WALL DETECTED")
    #             if Slow_oreint == "LEFT":
    #                 if angle < 0:
    #                     angle -= 0.1
    #             elif Slow_oreint == "RIGHT":
    #                 if angle > 0:
    #                     angle += 0.1
    #             else:
    #                 print("NO SLOW ORIENTATION")
    rc.drive.set_speed_angle(speed, angle)




def WALL_START_ID_3():
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
    
def update_slow():
    """Periodic updates for status information"""
    global ID, previous_ID
    print_current_markers()
    save_current_markers()
    previous_ID = ID

# Start the detector
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go() 