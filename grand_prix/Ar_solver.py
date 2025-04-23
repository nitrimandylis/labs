isSimulation = True
import math
import copy
import cv2 as cv
import numpy as np
# Remove problematic imports
# import matplotlib.pyplot as plt
# import ipywidgets as widgets
import statistics
from typing import Any, Tuple, List, Optional
from enum import Enum

# Import Racecar library
import sys
sys.path.append("../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar(True)

potential_colors = [
    ((10, 50, 50), (20, 255, 255),'ORANGE'),
    ((100, 150, 50), (110, 255, 255),'BLUE'),
    ((40, 50, 50), (80, 255, 255),'GREEN'),  # The HSV range for the color green
    ((170, 50, 50), (10, 255, 255),'RED'),
    ((110,59,50), (165,255,255),'PURPLE')
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
        
    # Gather raw AR marker data from ArUco
    try:
        aruco_data = cv.aruco.detectMarkers(
            image,
            cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250),
            parameters=cv.aruco.DetectorParameters_create()
        )
        
        # A list of ARMarker objects representing the AR markers found in aruco_data
        markers = []
        
        # Check if markers were detected
        if len(aruco_data[0]) > 0:
            for i in range(len(aruco_data[0])):
                corners = aruco_data[0][i][0].astype(np.int32)
                for j in range(len(corners)):
                    col = corners[j][0]
                    corners[j][0] = corners[j][1]
                    corners[j][1] = col
                marker_id = aruco_data[1][i][0]
                
                markers.append(ARMarker(marker_id, corners))
                markers[-1].detect_colors(image, potential_colors)
            
        return markers
    except Exception as e:
        print(f"Error detecting AR markers: {e}")
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

# Variables for car control
speed = 0.0
angle = 0.0
max_speed = 0.5
auto_mode = False

def start():
    """
    Initialize the car and set up variables
    """
    global speed, angle, max_speed, auto_mode
    
    # Initialize car control values
    speed = 0.0
    angle = 0.0
    auto_mode = False
    
    # Set maximum speed for safety
    rc.drive.set_max_speed(max_speed)
    
    # Print instructions to the console
    print("AR Marker Detector with Manual Control")
    print("--------------------------------------")
    print("Controls:")
    print("  Left trigger: Reverse")
    print("  Right trigger: Forward")
    print("  Left joystick: Steering")
    print("  A button: Display marker information")
    print("  B button: Toggle auto navigation mode")
    
    # Set update rate for slow update
    rc.set_update_slow_time(0.5)

def update():
    """
    Main update loop to handle car control and marker detection
    """
    global speed, angle, auto_mode
    
    # Get the current image
    image = rc.camera.get_color_image()
    
    # Toggle auto mode with B button
    if rc.controller.was_pressed(rc.controller.Button.B):
        auto_mode = not auto_mode
        print(f"Auto navigation mode: {'ON' if auto_mode else 'OFF'}")
    
    if auto_mode:
        # Auto navigation based on AR markers
        if image is not None:
            markers = get_ar_markers(image)
            if markers:
                # Take action based on the closest marker
                closest_marker = markers[0]  # For simplicity, just use the first marker
                marker_instruction = ar_info(closest_marker)
                print(f"Following instruction: {marker_instruction}")
                
                if "Turn Left" in marker_instruction:
                    angle = -0.5
                    speed = 0.3
                elif "Turn Right" in marker_instruction:
                    angle = 0.5
                    speed = 0.3
                elif "Lane Following" in marker_instruction:
                    # Center on the marker
                    marker_center = rc_utils.get_center_pixel(image)
                    if marker_center[1] < image.shape[1] // 2:
                        angle = -0.2
                    else:
                        angle = 0.2
                    speed = 0.3
                else:  # Default behavior including "Follow" or "Slalom"
                    angle = 0.0
                    speed = 0.3
            else:
                # No markers detected, slow down
                speed = 0.1
                # Keep the current angle
    else:
        # Manual control mode
        # Read controller inputs for manual car control
        left_trigger = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
        right_trigger = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
        left_joystick_x = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
        
        # Calculate speed based on triggers (right for forward, left for reverse)
        speed = right_trigger - left_trigger
        
        # Set steering angle based on left joystick
        angle = left_joystick_x
    
    # Display marker info when A button is pressed
    if rc.controller.was_pressed(rc.controller.Button.A):
        marker_info = get_markers_info()
        if marker_info:
            print("Detected markers:")
            for info in marker_info:
                print(f"- {info}")
        else:
            print("No markers detected")
    
    # Set the car's speed and angle
    rc.drive.set_speed_angle(speed, angle)
    
    # Display the image with AR marker overlay if available
    if image is not None:
        rc.display.show_color_image(image)

def update_slow():
    """
    Function for slower updates, like displaying status information
    """
    # Display current car status
    mode_str = "AUTO" if auto_mode else "MANUAL"
    print(f"Control mode: {mode_str}, Speed: {speed:.2f}, Angle: {angle:.2f}")
    
    # Try to detect markers periodically
    try:
        image = rc.camera.get_color_image()
        if image is not None:
            markers = get_ar_markers(image)
            if markers:
                print(f"Detected {len(markers)} AR markers")
                for marker in markers:
                    print(f"  - ID: {marker.get_id()}, Color: {marker.get_color()}")
    except Exception as e:
        print(f"Error in marker detection: {e}")

# Start the car
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()