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
import time
from typing import Any, Tuple, List, Optional
from enum import Enum, IntEnum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()
isSimulation = True

# Colors in HSV format
PURPLE = ((125, 100, 100), (140, 255, 255))
ORANGE = ((10, 150, 120), (25, 255, 255))
RED = ((170, 50, 50), (10, 255, 255))
BLUE = ((100, 150, 50), (110, 255, 255))
GREEN = ((40, 60, 60), (80, 255, 255))
YELLOW = ((20, 100, 100), (40, 255, 255))  # HSV range for yellow

# Contour detection
MIN_CONTOUR_AREA = 30
SHADOW_MAX_VALUE = 80    # Maximum value (brightness) for shadows
SHADOW_MAX_SATURATION = 60  # Maximum saturation for shadows
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# Line following color ranges with higher saturation to distinguish from shadows
COLORS = [
    ((0, 150, 120), (10, 255, 255)),   # Red color range
    ((40, 100, 100), (85, 255, 255)),  # Green color range
    ((90, 150, 120), (120, 255, 255))  # Blue color range
]

# LIDAR windows
LEFT_WINDOW = (-45, -15)
RIGHT_WINDOW = (15, 45)
FRONT_WINDOW = (-10, 10)  # Window for directly in front of the car

# Core control variables
contour_center = None
contour_area = 0
speed = 0.0
angle = 0.0
wall_following = False
ID = 0
COLOR = "none"

# Lane following variables
prioritylist = [PURPLE, ORANGE]
Lane_priority = 0  # Default to 0 (PURPLE priority)
largestcontour_center = None
secondcontour_center = None
generalcontour_center = None
accumulatedError = 0
lastError = 0

# Wall following variables
left_angle = 0
left_distance = 0
right_angle = 0
right_distance = 0
front_distance = 1000
cur_state = None  # Will be set in initialization
angle_to_yellow = 0  # Angle to the detected yellow object (-1.0 to 1.0)

# Marker tracking variables
previous_ID = -1   # Initialize with invalid marker ID
previous_colour = None
marker_timeout = 0
turning_timer = 0
current_time = 0  # Track current time since start
is_turning_right = False
distance_to_marker = 10000  # Initialize to a very large value
contour_corners = None      # Store marker corners
Slow_oreint = "NONE"        # Initialize orientation for slow turns
counter = 0
random_number = 1  # Initialize random_number variable for random seed

# Line tracking variables
previous_centers = []  # Store recent contour centers
prev_angle = 0  # Store previous angle for smoothing
target_speed = 0.8  # Default target speed (can be adjusted with controller)
last_update_time = 0  # Time of last update
recovery_mode = False  # Flag for recovery mode when tracking is lost
recovery_counter = 0   # Counter for recovery mode timing
last_good_angle = 0    # Last known good angle when tracking was reliable
min_speed_factor = 0.7  # Minimum speed factor (prevent going too slow)
current_color_index = -1  # Current color being tracked
debug_mode = True  # Enable to show shadow detection
only_show_priority = False  # Add global for display mode

# Define the potential colors for marker detection
potential_colors = [
    ((10, 50, 50), (20, 255, 255), 'ORANGE'),
    ((100, 150, 50), (110, 255, 255), 'BLUE'),
    ((40, 50, 50), (80, 255, 255), 'GREEN'),
    ((170, 50, 50), (10, 255, 255), 'RED'),
    ((110, 59, 50), (165, 255, 255), 'PURPLE')
]

########################################################################################
# Enums
########################################################################################

class State(IntEnum):
    LANE_FOLLOWING = 0
    LANE_FOLLOWING_ID_0 = 1
    LANE_FOLLOWING_ID_1 = 2
    WALL_FOLLOWING_ID_3 = 3
    MOVE = 0
    TURN = 1
    STOP = 2

class Orientation(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

########################################################################################
# Classes
########################################################################################

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

########################################################################################
# AR Marker Detection Functions
########################################################################################

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
    global ID, COLOR, distance_to_marker, contour_corners, Orientation, previous_ID
    
    image = rc.camera.get_color_image()
    if image is None:
        return False
        
    markers = get_ar_markers(image)
    if not markers:
        return False
        
    for marker in markers:
        # Store the current ID before updating
        current_ID = ID
        
        # Update marker information
        ID = marker.get_id()
        COLOR = marker.get_color()
        Orientation = marker.get_orientation()
        contour_corners = marker.get_corners()
        
        # Calculate distance to this marker
        distance_to_marker = calculate_marker_distance(contour_corners)
        
        # Only update previous_ID when we have a significant ID change
        # Don't count transition markers (199, 75) as significant changes
        if current_ID != ID and current_ID not in [75, 199, -1] and ID not in [75, 199]:
            previous_ID = current_ID
            print(f"Updated previous_ID from {current_ID} to {previous_ID}")
        
        print(f"ID: {ID}, COLOR: {COLOR}, Distance: {distance_to_marker:.1f}cm")
        return True  # Mark that we found a marker
        
    return False  # No marker found 

########################################################################################
# Utility Functions
########################################################################################

def preprocess_image_for_yellow(image):
    """
    Preprocesses the image to reduce shadow effects.
    """
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Apply a slight blur to reduce noise
    hsv = cv.GaussianBlur(hsv, (5, 5), 0)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the V channel
    # This helps normalize brightness across the image
    h, s, v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Enhance saturation slightly to make yellow pop more compared to shadows
    s = cv.multiply(s, 1.3)  # Increase saturation by 30%
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # Merge channels back
    hsv = cv.merge([h, s, v])
    
    return hsv

def create_shadow_mask(hsv_image):
    """
    Creates a binary mask that identifies shadow areas (low value and saturation).
    Returns a mask where 255 = shadow, 0 = not shadow
    """
    # Split HSV channels
    h, s, v = cv.split(hsv_image)
    
    # Create shadow mask where both saturation and value are low
    shadow_mask = cv.bitwise_and(
        cv.threshold(s, SHADOW_MAX_SATURATION, 255, cv.THRESH_BINARY_INV)[1],
        cv.threshold(v, SHADOW_MAX_VALUE, 255, cv.THRESH_BINARY_INV)[1]
    )
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv.morphologyEx(shadow_mask, cv.MORPH_OPEN, kernel)
    shadow_mask = cv.morphologyEx(shadow_mask, cv.MORPH_CLOSE, kernel)
    
    return shadow_mask

def preprocess_Line_image(image):
    """
    Preprocesses the image to reduce shadow effects.
    """
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Apply a slight blur to reduce noise
    hsv = cv.GaussianBlur(hsv, (5, 5), 0)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the V channel
    # This helps normalize brightness across the image
    h, s, v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Enhance saturation slightly to make colors pop more compared to shadows
    s = cv.multiply(s, 1.3)  # Increase saturation by 30%
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # Merge channels back
    hsv = cv.merge([h, s, v])
    
    return hsv

def create_shadow_mask_Line(hsv_image):
    """
    Creates a binary mask that identifies shadow areas (low value and saturation).
    Returns a mask where 255 = shadow, 0 = not shadow
    """
    # Split HSV channels
    h, s, v = cv.split(hsv_image)
    
    # Create shadow mask where both saturation and value are low
    shadow_mask = cv.bitwise_and(
        cv.threshold(s, SHADOW_MAX_SATURATION, 255, cv.THRESH_BINARY_INV)[1],
        cv.threshold(v, SHADOW_MAX_VALUE, 255, cv.THRESH_BINARY_INV)[1]
    )
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv.morphologyEx(shadow_mask, cv.MORPH_OPEN, kernel)
    shadow_mask = cv.morphologyEx(shadow_mask, cv.MORPH_CLOSE, kernel)
    
    return shadow_mask

def apply_shadow_overlay_line(image, shadow_mask, opacity=0.3):
    """
    Safely applies a shadow overlay to an image, handling size mismatches.
    
    Args:
        image: The base image to apply the overlay to
        shadow_mask: The shadow mask (single channel)
        opacity: The opacity of the overlay (0-1)
        
    Returns:
        The image with shadow overlay applied
    """
    try:
        # Make sure shadow mask has the same dimensions as the image
        if shadow_mask.shape[:2] != image.shape[:2]:
            shadow_mask = cv.resize(shadow_mask, (image.shape[1], image.shape[0]))
        
        # Convert mask to BGR for overlay
        shadow_overlay = cv.cvtColor(shadow_mask, cv.COLOR_GRAY2BGR)
        shadow_overlay[:, :, 0] = 0  # Set B channel to 0
        shadow_overlay[:, :, 1] = 0  # Set G channel to 0
        # Only keep R channel to show shadows as red
        
        # Blend the images using addWeighted
        return cv.addWeighted(image, 1.0, shadow_overlay, opacity, 0)
    except Exception as e:
        print(f"Warning: Failed to apply shadow overlay: {e}")
        return image  # Return original image if overlay fails 

########################################################################################
# Yellow Detection and Wall Following Functions
########################################################################################

def Can_see_yellow(min_contour_area=100):
    """
    Optimized version of yellow detection with reduced overhead.
    """
    global angle_to_yellow
    
    # Capture image once and check if it's valid
    image = rc.camera.get_color_image()
    if image is None:
        return False, None, 0
    
    # Create one focused crop region instead of multiple
    crop_bottom = int(rc.camera.get_height() * 2/3)
    crop_region = ((crop_bottom, 0), (rc.camera.get_height(), rc.camera.get_width()))
    
    # Crop the image
    cropped_image = rc_utils.crop(image, crop_region[0], crop_region[1])
    
    # Process image - simplified pipeline for better performance
    hsv = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)
    hsv = cv.GaussianBlur(hsv, (5, 5), 0)
    
    # Create color mask for yellow - skip shadow detection for yellow
    yellow_mask = cv.inRange(hsv, YELLOW[0], YELLOW[1])
    
    # Apply simple morphology to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv.morphologyEx(yellow_mask, cv.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by minimum area
    valid_contours = [c for c in contours if cv.contourArea(c) > min_contour_area]
    
    # If valid contours found, process the largest one
    if valid_contours:
        largest_contour = max(valid_contours, key=cv.contourArea)
        area = cv.contourArea(largest_contour)
        
        # Calculate center
        M = cv.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + crop_region[0][0]  # Adjust for crop offset
            
            # Calculate angle to yellow object (normalized from -1.0 to 1.0)
            image_center_x = rc.camera.get_width() / 2
            angle_to_yellow = (cx - image_center_x) / image_center_x
            angle_to_yellow = rc_utils.clamp(angle_to_yellow, -1.0, 1.0)
            
            # Only show visualization occasionally to reduce lag
            if int(time.time() * 2) % 6 == 0:  # Show every 3 seconds
                display_img = image.copy()
                adjusted_contour = largest_contour.copy()
                adjusted_contour[:, :, 1] += crop_region[0][0]
                cv.drawContours(display_img, [adjusted_contour], -1, (0, 255, 255), 2)
                cv.circle(display_img, (cx, cy), 5, (0, 0, 255), -1)
                rc.display.show_color_image(display_img)
            
            return True, (cy, cx), area
    
    # Reset angle if no yellow detected
    angle_to_yellow = 0
    return False, None, 0

def stop_WALL_FOLLOWING_ID_3():
    global speed
    global angle
    global cur_state
    global front_distance

    speed = 0
    angle = 0
    
    # If the path is clear again, start moving
    if front_distance > 40:
        cur_state = State.MOVE
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
        # Turn left - simplify angle calculation 
        angle = rc_utils.clamp(error / 25, -1.0, 1.0)
        
        # Apply fixed adjustments to make turns more responsive
        if -0.4 < angle < 0:
            angle -= 0.1
        elif angle < -0.4:
            angle -= 0.4
    else:   
        # Turn right - simplify angle calculation
        angle = rc_utils.clamp(error / 25, -1.0, 1.0)
        
        # Apply fixed adjustments to make turns more responsive
        if 0 < angle < 0.4:
            angle += 0.1
        elif angle > 0.4:
            angle += 0.4
            
    # Make sure angle stays within bounds
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    
    # Set appropriate speed for turning
    rc.drive.set_max_speed(0.28)

    # Override with stronger turn if obstacle directly ahead
    if front_distance < 30 and left_distance == right_distance:
        angle = 1

    # Exit turn state if walls are balanced
    if abs(error) < 10:
        cur_state = State.MOVE
    
    speed = 0.7

def move_WALL_FOLLOWING_ID_3():
    global speed
    global angle
    global left_distance
    global right_distance
    global front_distance
    global cur_state

    # Set default values for moving forward
    speed = 1
    angle = 0
    
    # Check if we need to turn based on wall distances
    if abs(left_distance-right_distance) > 10:
        cur_state = State.TURN

def WALL_FOLLOWING_UPDATE_ID_3():
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global front_distance
    global cur_state
    global speed
    global angle
    global random_number

    # Generate a random number but only occasionally
    if int(time.time()) % 5 == 0:  # Only update random number every 5 seconds
        random_number = random.randint(1, 100)
    
    # Get LIDAR samples once and reuse
    scan = rc.lidar.get_samples()
    left_angle, left_distance = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    front_angle, front_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    
    # If no point is detected in front, set front_distance to a very large value
    front_distance = 10000000 if front_dist is None else front_dist
    
    # Only print distance info occasionally to reduce console spam
    if int(time.time()) % 3 == 0:  # Every 3 seconds
        print(f"L: {left_distance:.0f}cm, R: {right_distance:.0f}cm, F: {front_distance:.0f}cm")
    
    # State machine
    if cur_state == State.MOVE:
        move_WALL_FOLLOWING_ID_3()
    elif cur_state == State.TURN:
        turn_WALL_FOLLOWING_ID_3()
    elif cur_state == State.STOP:
        stop_WALL_FOLLOWING_ID_3()
    
    # Set the final speed and angle
    if left_distance > 70 and right_distance > 70 and front_distance > 190:
        speed = 1
        rc.drive.set_max_speed(1)

        # Straighten gradually
        if angle > 0:
            angle -= 0.2
        elif angle < 0:
            angle += 0.2
    elif left_distance > 70 and right_distance > 70 and front_distance < 100:
        speed = 0.8
        rc.drive.set_max_speed(0.28)
        
    # Check for yellow once per call - store result to avoid duplicate calls
    yellow_result = Can_see_yellow()
    if yellow_result and yellow_result[1] is not None:
        speed = 1
        rc.drive.set_max_speed(0.35)
        if angle_to_yellow > 0:
            angle += 0.2
        elif angle_to_yellow < 0:
            angle -= 0.2
            
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    rc.drive.set_speed_angle(speed, angle)

def WALL_START_ID_3():
    rc.drive.stop()
    global cur_state
    global speed
    global angle
    global front_distance
    global random_number

    # Initialize front_distance to a large value
    front_distance = 10000000
    
    # Print start message
    print(">> Lab 4B - LIDAR Wall Following")
    rc.drive.set_max_speed(0.28)

########################################################################################
# Lane Following Functions
########################################################################################

def pid_Lane(Kp, Ki, Kd, target, current, dT):
    global accumulatedError
    global lastError
    error = target - current
    accumulatedError += error * dT
    accumulatedError = rc_utils.clamp(accumulatedError, -10, 10)
    deltaError = (error - lastError) / dT if dT > 0 else 0
    pTerm = Kp * error
    iTerm = Ki * accumulatedError
    dTerm = Kd * deltaError
    lastError = error
    return pTerm + iTerm + dTerm

def update_contour_Lane():
    global contour_center, contour_area
    global largestcontour_center, secondcontour_center, generalcontour_center
    global current_time

    image = rc.camera.get_color_image()
    
    if image is not None:
        crop_floor = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
        image = rc_utils.crop(image, crop_floor[0], crop_floor[1])
        
        largestcontour = None
        secondcontour = None
        largestcontour_center = (0, 0)
        secondcontour_center = (0, 0)
        generalcontour_center = (0, 0)
        
        for col in prioritylist:
            contours = rc_utils.find_contours(image, col[0], col[1])
            
            if not contours or len(contours) == 0:
                continue
                
            valid_contours = []
            for contour in contours:
                if contour is not None and cv.contourArea(contour) > MIN_CONTOUR_AREA:
                    valid_contours.append(contour)
            
            if len(valid_contours) == 0:
                continue
            
            sorted_contours = sorted(valid_contours, key=cv.contourArea, reverse=True)
            
            largestcontour = sorted_contours[0] if len(sorted_contours) > 0 else None
            secondcontour = sorted_contours[1] if len(sorted_contours) > 1 else None
            
            if largestcontour is not None:
                break
        
        if largestcontour is not None:
            largestcontour_center = rc_utils.get_contour_center(largestcontour)
            rc_utils.draw_contour(image, largestcontour, (0, 255, 0))
            if largestcontour_center is not None:
                rc_utils.draw_circle(image, largestcontour_center, (0, 255, 0))
                generalcontour_center = largestcontour_center
                contour_center = largestcontour_center
                contour_area = cv.contourArea(largestcontour)
        
        if secondcontour is not None:
            secondcontour_center = rc_utils.get_contour_center(secondcontour)
            rc_utils.draw_contour(image, secondcontour, (255, 0, 0))
            if secondcontour_center is not None:
                rc_utils.draw_circle(image, secondcontour_center, (255, 0, 0))
    
        rc.display.show_color_image(image)
    
    current_time += rc.get_delta_time()
    return largestcontour_center, secondcontour_center, generalcontour_center

def follow_two_lines_Lane():
    global speed, angle, last_good_angle, recovery_mode, recovery_counter, previous_centers
    
    cameraWidth = rc.camera.get_width()
    distancethreshold = 70
    delta_time = rc.get_delta_time()
    
    largestcontour_center, secondcontour_center, generalcontour_center = update_contour_Lane()
    
    has_largest = isinstance(largestcontour_center, tuple) and len(largestcontour_center) == 2 and largestcontour_center[0] != 0
    has_second = isinstance(secondcontour_center, tuple) and len(secondcontour_center) == 2 and secondcontour_center[0] != 0
    has_general = isinstance(generalcontour_center, tuple) and len(generalcontour_center) == 2 and generalcontour_center[0] != 0
    
    if has_largest and has_second:
        smallestx = min(largestcontour_center[1], secondcontour_center[1])
        largestx = max(largestcontour_center[1], secondcontour_center[1])
        center_point = (largestx + smallestx) / 2
        
        lane_width = largestx - smallestx
        print(f"Lane width: {lane_width} pixels")
        
        if (largestx - smallestx) > distancethreshold:
            target_point = center_point
            normalized_target = rc_utils.remap_range(target_point, 0, cameraWidth, -1, 1)
            angle = pid_Lane(0.5, 0.1, 0.2, 0, normalized_target, delta_time)
            speed = 0.3
        else:
            if center_point < (cameraWidth/2) - 30:
                angle = rc_utils.remap_range(center_point, 0, cameraWidth/2, 0.5, 0.1)
            elif center_point > (cameraWidth/2) + 30:
                angle = rc_utils.remap_range(center_point, cameraWidth/2, cameraWidth, -0.1, -0.5)
            speed = 0.2
        
        previous_centers.append((largestcontour_center[0], center_point))
        if len(previous_centers) > 4:
            previous_centers.pop(0)
            
        basic_angle = center_point - cameraWidth / 2
        basic_angle /= cameraWidth / 2
        last_good_angle = basic_angle
        
        recovery_mode = False
        recovery_counter = 0
            
    elif has_general:
        center_point = generalcontour_center[1]
        
        print(f"Following single line at x={center_point}")
        
        if center_point < cameraWidth/2:
            angle = rc_utils.remap_range(center_point, 0, cameraWidth/2, 0.5, 0.1)
        else:
            angle = rc_utils.remap_range(center_point, cameraWidth/2, cameraWidth, -0.1, -0.5)
        
        speed = 0.25
        
        previous_centers.append((generalcontour_center[0], center_point))
        if len(previous_centers) > 4:
            previous_centers.pop(0)
            
        basic_angle = center_point - cameraWidth / 2
        basic_angle /= cameraWidth / 2
        last_good_angle = basic_angle
        
        recovery_mode = False
        recovery_counter = 0
        
    else:
        enter_recovery_mode_Lane(delta_time)

def enter_recovery_mode_Lane(delta_time):
    global recovery_mode, recovery_counter, angle, speed, previous_centers
    
    if not recovery_mode:
        recovery_mode = True
        recovery_counter = 0
        print("Entering RECOVERY MODE - no lines detected")
    
    if len(previous_centers) > 0:
        if len(previous_centers) > 1:
            previous_centers.pop(0)
    
    recovery_counter += delta_time
    
    angle = last_good_angle
    
    search_amplitude = 0.3
    search_frequency = 1.5
    search_offset = math.sin(recovery_counter * search_frequency * 2 * math.pi) * search_amplitude
    angle += search_offset
    
    if int(recovery_counter * 10) % 5 == 0:
        print(f"RECOVERY MODE: Searching with angle {angle:.2f}")
    
    speed = 0.15
    
    if recovery_counter > 2.0:
        recovery_counter = 0

def start_Lane():
    global recovery_mode, recovery_counter, last_good_angle, previous_centers
    global accumulatedError, lastError, speed, angle, current_time
    
    accumulatedError = 0
    lastError = 0
    current_time = 0
    recovery_mode = False
    recovery_counter = 0
    last_good_angle = 0
    previous_centers = []
    speed = 0
    angle = 0
    
    rc.drive.set_speed_angle(0, 0)
    rc.drive.set_max_speed(0.5)
    
    print(
        ">> Lab F - Two Line Following Challenge\n"
        "\n"
        "Controls:\n"
        "   X button = set ORANGE as primary color\n"
        "   Y button = set PURPLE as primary color"
    )
    
    rc.set_update_slow_time(0.5)

def update_Lane():
    global prioritylist, angle, speed, Lane_priority
    
    # Set priority based on Lane_priority value
    if Lane_priority == 1:
        prioritylist = [ORANGE, PURPLE]
    else:  # Lane_priority == 0
        prioritylist = [PURPLE, ORANGE]
    
    follow_two_lines_Lane()
    
    if angle > 0:
        angle += 0.3
    elif angle < 0:
        angle -= 0.3
    angle = rc_utils.clamp(angle, -1, 1)
    
    speed_factor = 1.0 - abs(angle) * 1.5
    calculate_speed = speed * max(0.3, speed_factor)
    
    calculate_speed = rc_utils.clamp(calculate_speed, 0.1, 1)
    
    print(f"Speed: {calculate_speed:.2f}, Angle: {angle:.2f}")
    
    return calculate_speed, angle

def update_slow_Lane():
    global Lane_priority
    
    if rc.camera.get_color_image() is None:
        print("WARNING: No camera image available!")
    
    color_names = ["PURPLE", "ORANGE"]
    if Lane_priority == 0:
        print(f"Currently prioritizing: {color_names[0]}, then {color_names[1]}")
    else:  # Lane_priority == 1
        print(f"Currently prioritizing: {color_names[1]}, then {color_names[0]}")
    
    if recovery_mode:
        print(f"Mode: RECOVERY MODE (searching for {recovery_counter:.1f}s)")
    else:
        print("Mode: Two Line Following")
        
    print(f"Camera crop: Starting at 360px from top") 

########################################################################################
# Line Following Functions
########################################################################################

def update_contour_Line():
    """
    Updates the contour information based on the current camera image.
    """
    global contour_center, contour_area, previous_centers, recovery_mode, recovery_counter
    global last_good_angle, current_color_index

    # Capture the current camera image
    image = rc.camera.get_color_image()

    # If the image is None, reset the contour information
    if image is None:
        contour_center = None
        contour_area = 0
        return
    
    # Use multiple crop regions for more robust tracking
    crop_regions = []
    
    # Main crop region - dynamic based on speed
    if abs(speed) > 0.7:
        crop_y = 300  # Adjusted to not look too far ahead
    else:
        crop_y = 340  # Default - not as high up for better tracking
    
    # Add main crop region
    crop_regions.append(((crop_y, 0), (rc.camera.get_height(), rc.camera.get_width())))
    
    # If in recovery mode or at high speed, add a wider/lower crop region to help find the line
    if recovery_mode or abs(speed) > 0.6:
        crop_regions.append(((380, 0), (rc.camera.get_height(), rc.camera.get_width())))
    
    # Initialize variables for best contour
    best_contour = None
    best_contour_area = 0
    best_crop_index = 0
    best_color_index = -1
    shadow_mask = None
    
    # Try each crop region
    for i, crop_region in enumerate(crop_regions):
        # Crop the image with the current crop region
        cropped_image = rc_utils.crop(image, crop_region[0], crop_region[1])
        
        # Preprocess the image to reduce shadow effects
        preprocessed_image = preprocess_Line_image(cropped_image)
        
        # Create shadow mask
        shadow_mask = create_shadow_mask_Line(preprocessed_image)
        
        # Show shadow mask in debug mode
        if debug_mode and i == 0:
            debug_img = cropped_image.copy()
            debug_img = apply_shadow_overlay_line(debug_img, shadow_mask, 0.5)
            cv.putText(debug_img, "Red = Shadow Areas", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rc.display.show_color_image(debug_img)
        
        # Prioritize the current color if we're already tracking one
        color_indices = list(range(len(COLORS)))
        if current_color_index >= 0:
            if only_show_priority:
                # Only check the prioritized color
                color_indices = [current_color_index]
                print(f"ONLY checking {['Red', 'Green', 'Blue'][current_color_index]} color")
            else:
                # Still check all colors, but move prioritized color to front of list
                # and will give it a larger boost in the area calculation later
                color_indices.remove(current_color_index)
                color_indices.insert(0, current_color_index)
                
        # Print what we're checking in update_contour
        if i == 0 and (current_color_index >= 0 or recovery_mode):
            if current_color_index >= 0:
                mode_str = "EXCLUSIVE" if only_show_priority else "PRIORITY"
                print(f"Checking colors in {mode_str} mode: {[['Red', 'Green', 'Blue'][i] for i in color_indices]}")
            elif recovery_mode:
                print(f"In recovery mode... searching all colors")

        # Iterate over the color ranges and search for contours
        for idx in color_indices:
            testingColor = COLORS[idx]
            # Create a color mask directly using cv2 instead of rc_utils.create_mask

########################################################################################
# AR Marker Handler Functions
########################################################################################

def ID_1_Handler():
    """
    Handle ID 1 marker (lane following)
    """
    global ID, previous_ID, COLOR, Lane_priority
    
    # Save current marker data
    save_current_markers()
    
    # If marker 1 is far away, revert to previous behavior
    if ID == 1 and distance_to_marker > 32:
        # Only revert if we have a valid previous_ID
        if previous_ID not in [-1, 75, 199]:
            print(f"Reverting to previous_ID: {previous_ID}")
            ID = previous_ID
        else:
            print("No valid previous_ID to revert to")
    
    # If marker 1 is close enough, handle lane following behavior
    if ID == 1 and distance_to_marker < 32:
        if COLOR == "PURPLE":
            Lane_priority = 0
            speed, angle = update_Lane()
            rc.drive.set_speed_angle(speed, angle)
            print("Following PURPLE lane")
        if COLOR == "ORANGE":
            Lane_priority = 1
            speed, angle = update_Lane()
            rc.drive.set_speed_angle(speed, angle)
            print("Following ORANGE lane")

def Line_Handles_Color_ID():
    """
    Handle line following based on ID color
    """
    global COLOR, current_color_index, only_show_priority
    save_current_markers()
    if COLOR == "GREEN" or COLOR == "RED" or COLOR == "BLUE":
        # Set color priority based on detected color
        if COLOR == "GREEN":
            current_color_index = 1
        if COLOR == "RED":
            current_color_index = 0
        if COLOR == "BLUE":
            current_color_index = 2
            
        only_show_priority = True
        speed, angle, _ = lab_Line_compressed()
        rc.drive.set_speed_angle(speed, angle)

def ID_3_Handler():
    """
    Handle ID 3 marker (wall following)
    """
    if ID == 3:
        WALL_FOLLOWING_UPDATE_ID_3()

########################################################################################
# Main Control Functions
########################################################################################

def start():
    """
    Initialize the main program
    """
    global current_time, counter, Slow_oreint, previous_ID, ID, previous_colour, COLOR, cur_state
    print_info()
    
    # Initialize current_time
    current_time = 0
    counter = 0
    Slow_oreint = "NONE"
    previous_ID = -1  # Invalid ID to start
    ID = -1
    previous_colour = None
    COLOR = None
    cur_state = State.MOVE
    
    # Set update rate for slow update
    rc.set_update_slow_time(10)

def update():
    """
    Main update function - called on each frame
    """
    global previous_colour, Lane_priority, current_color_index, Slow_oreint
    global ID, previous_ID, marker_timeout, turning_timer, is_turning_right
    global contour_corners, distance_to_marker, current_time, counter, COLOR, Orientation
    
    # Update current time
    current_time += rc.get_delta_time()
    
    # Process markers only once per frame to avoid redundant processing
    current_image = rc.camera.get_color_image()
    if current_image is not None:
        # Get markers once and reuse the results
        markers = get_ar_markers(current_image)
        
        # Only do expensive highlighting if we found markers
        if markers:
            current_image = highlight_markers(current_image, markers)
            rc.display.show_color_image(current_image)
            
            # Update global variables based on first marker
            if len(markers) > 0:
                marker = markers[0]
                ID = marker.get_id()
                COLOR = marker.get_color()
                Orientation = marker.get_orientation()
                contour_corners = marker.get_corners()
                
                # Calculate distance to this marker
                distance_to_marker = calculate_marker_distance(contour_corners)
    
    # Only print marker info occasionally to reduce console output
    if int(current_time * 2) % 2 == 0:  # Every 0.5 seconds
        if ID != previous_ID:
            print(f"ID: {ID}, COLOR: {COLOR}, Distance: {distance_to_marker:.1f}cm")

    # Process marker logic - use optimized conditional structure
    print("previous_ID", previous_ID)
    if ID == 199 and COLOR is not None or ID == 0 and COLOR is not None:
        Line_Handles_Color_ID()
        print("lines")
    if ID == 1:
        ID_1_Handler()
        print("Lane")
    if ID == 3:
        ID_3_Handler()
        print("Wall")
    if ID == 199 and COLOR is None:
        # Handle marker 199 with wall following
        if distance_to_marker < 32:
            # Turn based on orientation
            if Orientation == Orientation.RIGHT:
                angle = 1  # Full right turn
                rc.drive.set_speed_angle(1, angle)
                counter += rc.get_delta_time()
                if previous_ID == 3 and counter > 0.2:
                    ID_3_Handler()
                    previous_ID = 3
                    ID = previous_ID
                if previous_ID == 1 and counter > 0.2:
                    ID_1_Handler()
                    previous_ID = 1
                    ID = previous_ID
                if previous_ID == 199 and previous_colour is not None:
                    ID = previous_ID
                    COLOR = previous_colour
                    Line_Handles_Color_ID()
            elif Orientation == Orientation.LEFT:
                angle = -1  # Full left turn
                rc.drive.set_speed_angle(1, angle)
                counter += rc.get_delta_time()
                
    if ID == 75:
        ID = previous_ID

def update_slow():
    """
    Slow update function - called at intervals
    """
    global ID, previous_ID, COLOR, previous_colour
    
    # Print current status
    print_current_markers()
    
    # Save marker information
    found_marker = save_current_markers()
    
    # Only update previous values if the current ID is important (not a transition marker)
    if ID not in [75, 199]:
        if ID != previous_ID and previous_ID != -1:
            print(f"Slow Update: ID changed from {previous_ID} to {ID}")
        
        # Store previous values for the next update
        previous_colour = COLOR
    
    print(f"Current IDs: ID={ID}, previous_ID={previous_ID}, COLOR={COLOR}, previous_colour={previous_colour}")

########################################################################################
# Program Entry Point
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()