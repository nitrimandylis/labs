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

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum
from enum import Enum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

MIN_CONTOUR_AREA = 30

PURPLE = ((125, 100, 100), (145, 255, 255))
ORANGE = ((5, 100, 100), (25, 255, 255))

contour_center = None
contour_area = 0
speed = 0
angle = 0
prioritylist = [PURPLE, ORANGE]
Lane_priority = 0  # Default to 0 (PURPLE priority)
previous_ID = -1   # Initialize previous_ID with an invalid marker ID

recovery_mode = False
recovery_counter = 0
last_good_angle = 0
previous_centers = []
current_time = 0

largestcontour_center = None
secondcontour_center = None
generalcontour_center = None

accumulatedError = 0
lastError = 0

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
GREEN = ((60, 200, 200), (80, 255, 255))  # Neon green has higher saturation and value
ORANGE = ((5, 100, 100), (25, 255, 255))
PURPLE = ((125, 100, 100), (145, 255, 255))
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
random_number = 1  # Initialize random_number variable for random seed
angle_to_yellow = 0  # Angle to the detected yellow object (-1.0 to 1.0)
angle_to_marker = 0  # Angle to the detected marker (-1.0 to 1.0)

# Updated YELLOW color range with wider values for better detection
YELLOW = ((20, 100, 100), (40, 255, 255))  # HSV range for yellow

# Shadow detection constants from lab_f.py
SHADOW_MAX_VALUE = 80    # Maximum value (brightness) for shadows
SHADOW_MAX_SATURATION = 60  # Maximum saturation for shadows

MIN_CONTOUR_AREA = 30

# Define the crop region for the camera image, which is used to focus on the floor in front of the racecar
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# Define a list of color ranges to search for in the camera image with much higher minimum saturation
# to better distinguish from shadows
COLORS = [
    # Red color range - much higher saturation minimum
    ((0, 150, 120), (10, 255, 255)),   
    # Neon Green color range - very high saturation and value
    ((60, 200, 200), (80, 255, 255)),  
    # Blue color range - much higher saturation minimum
    ((90, 150, 120), (120, 255, 255)),
    # Orange color range
    ((5, 100, 100), (25, 255, 255)),
    # Purple color range
    ((125, 100, 100), (145, 255, 255))
]

# Shadow detection constants
SHADOW_MAX_VALUE = 80    # Maximum value (brightness) for shadows
SHADOW_MAX_SATURATION = 60  # Maximum saturation for shadows

# Initialize variables to store the speed, angle, contour center, and contour area
speed = 0.0  
angle = 0.0  
contour_center = None  
contour_area = 0  

# New globals for high-speed improvements
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

# Add these color ranges near the other color definitions
SLALOM_RED = ((160, 0, 0), (179, 255, 255))
SLALOM_BLUE = ((90, 120, 120), (120, 255, 255))

# Add this enum definition with the other enums
class SlalomState(Enum):
    """
    Enum class representing different states of the slalom behavior.
    """
    search = 0
    red = 1
    blue = 2
    linear = 3

# Add these globals with the other globals
slalom_state = SlalomState.search
slalom_color_priority = "RED"
slalom_last_distance = 0
slalom_counter = 0

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

speed = 1.0
angle = 0.0
left_angle = 0
left_distance = 0
right_angle = 0
right_distance = 0
front_distance = 100

potential_colors = [
    ((5, 100, 100), (25, 255, 255),'ORANGE'),
    ((100, 150, 50), (110, 255, 255),'BLUE'),
    ((60, 200, 200), (80, 255, 255),'GREEN'),  # The HSV range for neon green
    ((170, 50, 50), (10, 255, 255),'RED'),
    ((125, 100, 100), (145, 255, 255),'PURPLE')
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
        self.__color = None
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
        
        # Calculate angle to marker
        marker_angle = calculate_angle_to_marker(corners)
        
        # Display marker info
        marker_info = f"ID: {marker.get_id()}, Color: {marker.get_color()}"
        angle_info = f"Angle: {marker_angle:.2f}"
        instruction = ar_info(marker)
        
        # Add text with a background for readability
        cv.putText(marked_image, marker_info, (center_x - 10, center_y - 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv.putText(marked_image, marker_info, (center_x - 10, center_y - 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv.putText(marked_image, angle_info, (center_x - 10, center_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv.putText(marked_image, angle_info, (center_x - 10, center_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv.putText(marked_image, instruction, (center_x - 10, center_y + 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv.putText(marked_image, instruction, (center_x - 10, center_y + 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return marked_image

def print_info():
    pass

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

def calculate_angle_to_marker(marker_corners):
    """
    Calculate the angle to a marker (normalized from -1.0 to 1.0)
    
    Args:
        marker_corners: The corners of the marker
        
    Returns:
        float: Normalized angle to marker (-1.0 = far left, 0 = center, 1.0 = far right)
    """
    if marker_corners is None:
        return 0
        
    # Calculate center of marker
    x_coords = [corner[1] for corner in marker_corners]
    y_coords = [corner[0] for corner in marker_corners]
    
    center_x = sum(x_coords) / len(x_coords)
    
    # Calculate angle (normalized from -1.0 to 1.0)
    image_center_x = rc.camera.get_width() / 2
    angle_to_marker = (center_x - image_center_x) / image_center_x
    
    return rc_utils.clamp(angle_to_marker, -1.0, 1.0)


    """
    Detect potential AR markers in the camera view, including partial markers.
    Can detect and return information even when only part of a marker is visible.
    
    Returns:
        tuple: (distance_to_marker, angle_to_marker, marker_id)
              If no marker (full or partial) is detected, returns (10000, 0, -1)
    """
    # Get current camera image
    image = rc.camera.get_color_image()
    if image is None:
        return 10000, 0, -1
    
    display_img = None
    show_debug = int(time.time()) % 3 == 0  # Only show debug visuals every 3 seconds
    
    if show_debug:
        display_img = image.copy()
    
    # STEP 1: Try standard AR marker detection first
    markers = get_ar_markers(image)
    if markers:
        # Full marker detected - use standard processing
        marker = markers[0]
        marker_id = marker.get_id()
        marker_corners = marker.get_corners()
        
        # Calculate distance and angle
        distance = calculate_marker_distance(marker_corners)
        angle = calculate_angle_to_marker(marker_corners)
        
        if show_debug and display_img is not None:
            # Draw full marker
            corners = marker.get_corners()
            cv.line(display_img, (corners[0][1], corners[0][0]), (corners[1][1], corners[1][0]), (0, 255, 0), 2)
            cv.line(display_img, (corners[1][1], corners[1][0]), (corners[2][1], corners[2][0]), (0, 255, 0), 2)
            cv.line(display_img, (corners[2][1], corners[2][0]), (corners[3][1], corners[3][0]), (0, 255, 0), 2)
            cv.line(display_img, (corners[3][1], corners[3][0]), (corners[0][1], corners[0][0]), (0, 255, 0), 2)
            
            # Add marker info text
            cv.putText(display_img, f"Full Marker: ID={marker_id}, Dist={distance:.1f}cm", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(display_img, f"Angle: {angle:.2f}", 
                      (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            rc.display.show_color_image(display_img)
        
        return distance, angle, marker_id
    
    # STEP 2: No full markers detected, look for partial markers
    # Convert to grayscale for better edge detection
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detector
    edges = cv.Canny(blurred, 50, 200)
    
    # Find contours in the edge image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that could be AR markers (square/rectangle-like)
    possible_marker_contours = []
    for contour in contours:
        # Skip tiny contours
        if cv.contourArea(contour) < 1000:
            continue
        
        # Approximate the contour to simplify it
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.03 * perimeter, True)
        
        # Check if it has 4 corners (rectangle/square) or is close to it (3-6 corners)
        if 3 <= len(approx) <= 6:
            # Calculate rectangularity - how rectangular is this shape?
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            rect_area = cv.contourArea(box)
            contour_area = cv.contourArea(contour)
            
            # If contour area is close to its bounding rectangle area, it's likely rectangular
            if rect_area > 0 and contour_area / rect_area > 0.7:
                possible_marker_contours.append(approx)
    
    # If no potential markers found, return default values
    if not possible_marker_contours:
        if show_debug and display_img is not None:
            cv.putText(display_img, "No markers detected", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rc.display.show_color_image(display_img)
        return 10000, 0, -1
    
    # Find the largest potential marker contour
    best_contour = max(possible_marker_contours, key=cv.contourArea)
    
    # Get the center of the contour
    M = cv.moments(best_contour)
    if M["m00"] == 0:
        return 10000, 0, -1
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Calculate approximate distance based on contour area
    # This is an approximation - actual distance would require camera calibration
    contour_area = cv.contourArea(best_contour)
    estimated_distance = 100000 / max(contour_area, 1)  # Similar formula to calculate_marker_distance
    
    # Calculate angle to the contour center
    image_center_x = rc.camera.get_width() / 2
    angle = (cx - image_center_x) / image_center_x
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    
    if show_debug and display_img is not None:
        # Draw the contour
        cv.drawContours(display_img, [best_contour], 0, (0, 255, 255), 2)
        
        # Draw center point
        cv.circle(display_img, (cx, cy), 5, (0, 0, 255), -1)
        
        # Add info text
        cv.putText(display_img, f"Partial Marker: Dist≈{estimated_distance:.1f}cm", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(display_img, f"Angle: {angle:.2f}", 
                  (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        rc.display.show_color_image(display_img)
    
    # Return the estimated values with ID=-1 to indicate it's a partial marker
    return estimated_distance, angle, -1

def save_current_markers():
    global ID, COLOR, distance_to_marker, contour_corners, Orientation, previous_ID, angle_to_marker
    
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
        
        # Calculate distance and angle to this marker
        distance_to_marker = calculate_marker_distance(contour_corners)
        angle_to_marker = calculate_angle_to_marker(contour_corners)
        
        # Only update previous_ID when we have a significant ID change
        # Don't count transition markers (199, 75) as significant changes
        if current_ID != ID and current_ID not in [75, 199, -1] and ID not in [75, 199]:
            previous_ID = current_ID
            print(f"Updated previous_ID from {current_ID} to {previous_ID}")
        
        print(f"ID: {ID}, COLOR: {COLOR}, Distance: {distance_to_marker:.1f}cm, Angle: {angle_to_marker:.2f}")
        return True  # Mark that we found a marker
        
    return False  # No marker found

def start():
    """Initialize the marker detector"""
    global current_time, counter, Slow_oreint, previous_ID, ID, previous_colour, COLOR, angle_to_marker
    global slalom_state, slalom_color_priority, slalom_last_distance, slalom_counter
    
    print_info()
    
    # Initialize current_time
    current_time = 0
    counter = 0
    Slow_oreint = "NONE"
    previous_ID = -1  # Invalid ID to start
    ID = 3
    previous_colour = None
    COLOR = None
    angle_to_marker = 0
    WALL_START_ID_3()
    start_Lane()
    start_line()
    # Initialize slalom variables
    slalom_state = SlalomState.search
    slalom_color_priority = "RED"
    slalom_last_distance = 0
    slalom_counter = 0
    
    # Set update rate for slow update
    rc.set_update_slow_time(10)

def ID_1_Handler():
    global ID, previous_ID, COLOR, Lane_priority, previous_colour, angle_to_marker, angle
    

    # Save current marker data
    save_current_markers()
    distance_to_marker = calculate_marker_distance(contour_corners)
    previous_ID = 1
    if COLOR is not None:
        previous_colour = COLOR

    # If marker 1 is far away, revert to previous behavior
    if ID == 1 and distance_to_marker > 32:
        # Only revert if we dhave a valid previous_ID
        print("Marker not close enough for lane following")
        
        # Add wall avoidance while driving toward the marker
        lidar_data = rc.lidar.get_samples()
        if lidar_data is not None and len(lidar_data) > 0:
            min_distance = np.min(lidar_data)
            min_angle = np.argmin(lidar_data)
            
            # If obstacle detected, adjust steering to avoid it
            if min_distance < 100 and abs(min_angle - 0) < 60:
                # Calculate avoidance angle (turn away from obstacle)
                avoidance_angle = 0.5 if min_angle < 180 else -0.5
                # Blend marker approach with obstacle avoidance
                if angle_to_marker > 0:
                    final_angle = angle_to_marker - avoidance_angle
                elif angle_to_marker < 0 :
                    final_angle = angle_to_marker + avoidance_angle

            
                final_angle = rc_utils.clamp(final_angle, -1.0, 1.0)
                rc.drive.set_speed_angle(0.3, final_angle)
            else:
                # No obstacle, continue toward marker
                rc.drive.set_speed_angle(0.8, angle_to_marker)
        else:
            # No valid lidar data, continue toward marker
            rc.drive.set_speed_angle(1, angle_to_marker)
        
        # if previous_ID == -1 or previous_ID == None:
        #     previous_ID = 1   
    # If marker 1 is close enough, handle lane following behavior
    if ID == 1 and distance_to_marker < 32 or distance_to_marker == None and ID == 1 or distance_to_marker == None and ID == 1 or ID == 2:
        print("Lane following activated colour not set yet")
        if COLOR == "PURPLE" or previous_colour == "PURPLE":
            Lane_priority = 0
            speed, angle = update_Lane()
            rc.drive.set_speed_angle(speed, angle)
            print("Following PURPLE lane")
        if COLOR == "ORANGE" or previous_colour == "ORANGE":
            Lane_priority = 1
            speed, angle = update_Lane()
            rc.drive.set_speed_angle(speed, angle)
            print("Following ORANGE lane")

def Line_Handles_Color_ID():
    global COLOR, current_color_index , only_show_priority , previous_colour, previous_ID
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
        previous_colour = COLOR
        speed, angle, _ = lab_Line_compressed()
        rc.drive.set_speed_angle(angle, speed)
    if ID == 1 and previous_colour == "GREEN" or ID == 1 and previous_colour == "RED" or ID == 1 and previous_colour == "BLUE":
        if previous_colour == "GREEN":
            current_color_index = 1
        if previous_colour == "RED":
            current_color_index = 0
        if previous_colour == "BLUE":
            current_color_index = 2
        #possible bug here to fix add previous_colour = previous_colour    
        only_show_priority = True
        previous_ID = 1
        speed, angle, _ = lab_Line_compressed()
        rc.drive.set_speed_angle(angle, speed)
def ID_3_Handler():
    global previous_ID ,distance_to_marker , angle_to_marker
    if ID == 3:
        WALL_FOLLOWING_UPDATE_ID_3()
        previous_ID = 3

def ID_2_Handler():
    global previous_ID , distance_to_marker , angle_to_marker 
    
    print("PREVIOUSE id WHEN id 2 :    ", previous_ID)
    print("distance to marker when ID2 :", distance_to_marker)
    if distance_to_marker > 30 and distance_to_marker != 10000:
        print("Marker not close enough for cone slalom")
        if previous_ID == 1:
            ID_1_Handler()
        elif previous_ID == 3:
            ID_3_Handler()
        else:
            rc.drive.set_speed_angle(0.6, angle_to_marker)
            
        
        print("More than 10")
    if distance_to_marker <= 30 and ID == 2 or distance_to_marker == None and ID == 2 or distance_to_marker == 10000 :
        previous_ID = 2
        ID_2_Updtae()
        if detec_change() == True:
            angle = ID_2_Updtae()
            angle += 0.4
            rc.drive.set_speed_angle(0.7, angle)
    

  


def update():
    """Main update function for marker detection"""
    global previous_colour, Lane_priority, current_color_index, Slow_oreint, ID, previous_ID, marker_timeout, turning_timer, is_turning_right, contour_corners, distance_to_marker, current_time , counter, COLOR , Orientation
    print("id IS :", ID)
    print("previous ID IS :", previous_ID)
    print("COLOR IS :", COLOR)
    print("previous COLOUR IS :", previous_colour)
    # Update current time
    current_time += rc.get_delta_time()
    save_current_markers()
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
                detected_id = marker.get_id()
                
                # Only accept valid IDs
                if detected_id in [0, 1, 2, 3, 199]:
                    ID = detected_id
                else:
                    # Invalid ID detected, keep using previous_ID
                    print(f"Invalid ID {detected_id} detected, using previous ID {previous_ID}")
                    ID = previous_ID
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
    if ID == 199 and COLOR is not None or ID == 0 and COLOR is not None:
        Line_Handles_Color_ID()
        print("lines")
    if ID == 1 :
        ID_1_Handler()
        print("Lane")
    if ID == 2:
        ID_2_Handler()
        print("Slalom")
    if ID == 3:
        ID_3_Handler()
        print("Wall")
    if ID == 4:
        rc.drive.set_max_speed(1)
        if COLOR == "RED" or distance_to_marker < 35:
            rc.drive.set_speed_angle(0, angle_to_marker)
        elif COLOR == "BLUE":
            rc.drive.set_speed_angle(1,angle_to_marker)
        elif COLOR == "ORANGE":
            rc.drive.set_speed_angle(1,angle_to_marker)
        elif COLOR == "RED" and distance_to_marker < 70:
            rc.drive.set_speed_angle(0.8,angle_to_marker)
        
        
    if ID not in [0,1,2,3,4,199]:
        ID = previous_ID

    if ID == 199 and COLOR is None:
        print("Precious ID at ID 199 is:", previous_ID)

        rc.drive.set_speed_angle(0.5, angle_to_marker)
        if distance_to_marker < 20 and previous_ID == 1 or distance_to_marker <= 0 and previous_ID == 1:
            print("Lane Following ID 199 reaction")
            counter += rc.get_delta_time()
            if Orientation == Orientation.RIGHT:
                angle = 0.8
            elif Orientation == Orientation.LEFT:
                angle = -0.8
            else:
                angle = 0
            rc.drive.set_speed_angle(0.3, angle)
            if counter > 0.2:
                print("Lane Activated 199")
                ID_1_Handler()
                rc.drive.set_speed_angle(1, angle)
                previous_ID = 1
                ID = previous_ID
                    

        if distance_to_marker < 32 and ID != 1:
            print("Other ID 199 reaction")
            angle = -1  # Full left turn
            
            counter += rc.get_delta_time()
            if Orientation == Orientation.RIGHT:
                angle = 1
            elif Orientation == Orientation.LEFT:
                angle = -1
            else:
                angle = 0

            rc.drive.set_speed_angle(1, angle)
            
            if previous_ID == 3 and counter > 0.2:
                ID_3_Handler()
                previous_ID = 3
                ID = previous_ID
                
            if previous_ID == 2 and counter > 0.2:
                ID_2_Handler()
                previous_ID = 2
                ID = previous_ID



            # If no special handling needed, continue with normal behavior
            
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


def turn_WALL_FOLLOWING_ID_3():
    """
    Turn the car to balance distances to walls on both sides
    """
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global front_distance
    global angle
    global cur_state
    global speed

    error = right_distance - left_distance
    if error < 0:
        # Turn left - simplify angle calculation 
        angle = rc_utils.remap_range(error, -15, 15, -1, 1)
        

    else:   
        # Turn right - simplify angle calculation
        angle = rc_utils.remap_range(error, -15, 15, -1, 1)
        
    # if angle > 0 :
    #     angle += 0.1
    # elif angle < 0 :
    #     angle -= 0.1
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    
    # Set appropriate speed for turning
    rc.drive.set_max_speed(0.28)

    # Override with stronger turn if obstacle directly ahead
    if front_distance < 10 and left_distance != right_distance:
        angle = 0.7

    # Exit turn state if walls are balanced
    if abs(error) < 10:
        cur_state = State.move
    
    speed = 1


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
        cur_state = State.turn

def WALL_FOLLOWING_UPDATE_ID_3():
    global left_angle
    global left_distance
    global right_angle
    global right_distance
    global front_distance
    global cur_state
    global speed
    global angle
    global counter
    global temp_distandce_front
    global temp_counter
    
    
    # Get LIDAR samples once and reuse
    scan = rc.lidar.get_samples()
    left_angle, left_distance = rc_utils.get_lidar_closest_point(LEFT_WINDOW,scan)
    right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    front_angle, front_dist = rc_utils.get_lidar_closest_point(FRONT_WINDOW, scan)
    
    # If no point is detected in front, set front_distance to a very large value
    front_distance = 10000000 if front_dist is None else front_dist
    
    # Only print distance info occasionally to reduce console spam
    if int(time.time()) % 3 == 0:  # Every 3 seconds
        print(f"L: {left_distance:.0f}cm, R: {right_distance:.0f}cm, F: {front_distance:.0f}cm")
    
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
        rc.drive.set_max_speed(0.5)

        # Straighten gradually

    elif left_distance > 70 and right_distance > 70 and front_distance < 100:
        speed = 0.8
        rc.drive.set_max_speed(0.28)
    counter += rc.get_delta_time()
    if int(time.time()) % 10000 == 0:
        WALL_START_ID_3()
    if int(time.time()) % 2 == 0:
        print("temp values updated")
        temp_distandce_front = front_distance
        temp_counter = counter
    


    if temp_distandce_front == front_distance and counter - 6 == temp_counter:
        rc.drive.set_speed_angle(-0.5, 0)
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
    global temp_distandce_front
    global temp_counter
    
    temp_distandce_front  = 0
    temp_counter = 10

    # Initialize front_distance to a large value
    front_distance = 100000
    
    # Print start message
    print(">> Lab 4B - LIDAR Wall Following")
    rc.drive.set_max_speed(0.28)
    
def update_slow():
    """Periodic updates for status information"""
    global ID, previous_ID, COLOR, previous_colour , angle_to_marker

    update_slow_Lane()
    update_slow_line()
    # Print current status
    print_current_markers()
    angle_to_marker = 0
    
    # Save marker information
    found_marker = save_current_markers()
    
    # Only update previous values if the current ID is important (not a transition marker)
    if COLOR is not None:
        previous_colour = COLOR
    
    print(f"Current IDs: ID={ID}, previous_ID={previous_ID}, COLOR={COLOR}, previous_colour={previous_colour}")

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
              # Show every 3 seconds
            display_img = image.copy()
            adjusted_contour = largest_contour.copy()
            adjusted_contour[:, :, 1] += crop_region[0][0]
            cv.drawContours(display_img, [adjusted_contour], -1, (0, 255, 255), 2)
            cv.circle(display_img, (cx, cy), 5, (0, 0, 255), -1)
            rc.display.show_color_image(display_img)
            
            return True, (cy, cx), area
    
    # Reset angle if no yellow detected
    angle_to_yellow = 2
    return False, None, 1

def detec_change(min_variance_threshold=50, texture_threshold=15):
    """
    Detects changes in the surface texture or color, particularly focusing on gray asphalt surfaces.
    More relaxed detection parameters to trigger True more easily.
    
    Args:
        min_variance_threshold: Minimum threshold for variance to detect texture change (lowered)
        texture_threshold: Threshold for texture gradient detection (lowered)
        
    Returns:
        tuple: (change_detected, change_position, change_direction)
            - change_detected: Boolean indicating if a change was detected
            - change_position: (row, col) position of the detected change
            - change_direction: "left", "right", "ahead", or None if no change detected
    """
    # Get the current camera image
    image = rc.camera.get_color_image()
    if image is None:
        return False, None, None
    
    # Focus on the lower part of the image where the road surface is visible
    crop_height = int(rc.camera.get_height() * 0.3)  # Start at 30% down the image
    crop_region = ((crop_height, 0), (rc.camera.get_height(), rc.camera.get_width()))
    cropped_image = rc_utils.crop(image, crop_region[0], crop_region[1])
    
    # Convert to grayscale for texture analysis
    gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate standard deviation for different regions to detect texture changes
    height, width = blurred.shape
    left_region = blurred[:, 0:width//3]
    center_region = blurred[:, width//3:2*width//3]
    right_region = blurred[:, 2*width//3:width]
    
    left_std = np.std(left_region)
    center_std = np.std(center_region)
    right_std = np.std(right_region)
    
    # Calculate texture gradient using Sobel operator (simplified)
    sobel_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Calculate average gradient for each region
    left_gradient = np.mean(gradient_magnitude[:, 0:width//3])
    center_gradient = np.mean(gradient_magnitude[:, width//3:2*width//3])
    right_gradient = np.mean(gradient_magnitude[:, 2*width//3:width])
    
    # Calculate the average gray level for each region
    left_gray = np.mean(left_region)
    center_gray = np.mean(center_region)
    right_gray = np.mean(right_region)
    
    # Debug visualization
    show_debug = int(time.time() * 2) % 12 == 0  # Show every 6 seconds to reduce performance impact
    if show_debug:
        display_img = cropped_image.copy()
        
        # Add text with region stats
        cv.putText(display_img, f"L: {left_std:.1f}, {left_gray:.1f}", (10, 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv.putText(display_img, f"C: {center_std:.1f}, {center_gray:.1f}", (width//3 + 10, 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv.putText(display_img, f"R: {right_std:.1f}, {right_gray:.1f}", (2*width//3 + 10, 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw region boundaries
        cv.line(display_img, (width//3, 0), (width//3, height), (0, 255, 0), 1)
        cv.line(display_img, (2*width//3, 0), (2*width//3, height), (0, 255, 0), 1)
        
        rc.display.show_color_image(display_img)
    
    # Determine if there's a significant change in the texture or color
    change_detected = False
    change_position = None
    change_direction = None
    
    # Calculate pairwise differences between regions (more sensitive detection)
    left_center_std_diff = abs(left_std - center_std)
    center_right_std_diff = abs(center_std - right_std)
    left_right_std_diff = abs(left_std - right_std)
    
    left_center_gray_diff = abs(left_gray - center_gray)
    center_right_gray_diff = abs(center_gray - right_gray)
    left_right_gray_diff = abs(left_gray - right_gray)
    
    left_center_gradient_diff = abs(left_gradient - center_gradient)
    center_right_gradient_diff = abs(center_gradient - right_gradient)
    left_right_gradient_diff = abs(left_gradient - right_gradient)
    
    # More relaxed detection logic - any significant difference will trigger detection
    # Check if ANY of the differences exceed the lowered thresholds
    if (left_center_std_diff > min_variance_threshold/2 or 
        center_right_std_diff > min_variance_threshold/2 or
        left_right_std_diff > min_variance_threshold/2 or
        left_center_gray_diff > min_variance_threshold/1.5 or
        center_right_gray_diff > min_variance_threshold/1.5 or
        left_right_gray_diff > min_variance_threshold/1.5 or
        left_center_gradient_diff > texture_threshold/1.5 or
        center_right_gradient_diff > texture_threshold/1.5 or
        left_right_gradient_diff > texture_threshold/1.5):
        
        change_detected = True
        
        # Determine change region based on which side shows the most different texture/color
        # Look for the region with the most different standard deviation compared to others
        if (left_std < center_std and left_std < right_std) or (left_gray < center_gray and left_gray < right_gray):
            # Left region is different (smoother/different color)
            change_direction = "left"
            change_position = (height//2, width//6)
        elif (right_std < center_std and right_std < left_std) or (right_gray < center_gray and right_gray < left_gray):
            # Right region is different (smoother/different color)
            change_direction = "right"
            change_position = (height//2, 5*width//6)
        else:
            # Center is different, or ambiguous change
            change_direction = "ahead"
            change_position = (height//2, width//2)
    
    return change_detected, change_position, change_direction

def lab_Line_compressed():
    """
    Updates the racecar's speed and angle based on the current contour information.

    This function updates the contour information, and then calculates the new speed
    and angle based on the contour center and area. It also checks for controller input
    to adjust the speed and angle.
    """
    global speed
    global angle
    global prev_angle
    global target_speed
    global recovery_mode
    global recovery_counter
    global last_good_angle
    global last_update_time
    global debug_mode
    global current_color_index
    global only_show_priority
    global angle_to_marker

    # Calculate delta time for smoother motion and timing
    current_time = rc.get_delta_time()
    delta_time = current_time - last_update_time
    last_update_time = current_time
    rc.drive.set_max_speed(1)

    # Update the contour information
    update_contour_Line()

    # Speed control with controller
    new_angle = 0
    
    if contour_center is not None:
        # Normal tracking mode - contour found
        # Basic angle calculation
        new_angle = contour_center[1] - rc.camera.get_width() / 2
        new_angle /= rc.camera.get_width() / 2

        # Add predictive steering if we have enough history
        if len(previous_centers) >= 3:
            # Calculate movement vector to predict where contour is heading
            # Use first and last points for a better estimate of direction
            dx = previous_centers[-1][1] - previous_centers[0][1]
            
            # Add prediction factor to steering (stronger at higher speeds)
            prediction_factor = 0.4 * abs(speed)  # Reduced from 0.6 for less aggressive prediction
            predicted_x = contour_center[1] + (dx * prediction_factor)
            predicted_angle = predicted_x - rc.camera.get_width() / 2
            predicted_angle /= rc.camera.get_width() / 2
            
            # Blend current position with prediction - more weight to prediction at high speeds
            prediction_weight = min(0.4, 0.2 + (abs(speed) * 0.2))  # Reduced from 0.5 max
            new_angle = ((1 - prediction_weight) * new_angle) + (prediction_weight * predicted_angle)
    
    elif recovery_mode:
        recovery_counter += delta_time
        
        # Start with the last known good angle
        new_angle = last_good_angle
        
        # Add a sinusoidal search pattern to try to find the line again
        search_amplitude = 0  # How far to search left/right
        search_frequency = 1.5  # How fast to search (increased from 1.0)
        search_offset = math.sin(recovery_counter * search_frequency * 2 * math.pi) * search_amplitude
        new_angle += search_offset
        
        # Don't print every frame to reduce console spam
        if int(recovery_counter * 10) % 5 == 0:
            print(f"RECOVERY MODE: Searching with angle {new_angle:.2f}")
    
    # Apply speed-dependent steering sensitivity (less reduction)
    if abs(speed) > 1:
        # Reduce steering sensitivity at high speeds to prevent oscillation
        new_angle *= 0.65  # Less reduction (was 0.65)
    elif abs(speed) > 1:
        new_angle *= 0.85  # Less reduction (was 0.8)
    
    # Calculate target raw speed - start with full target speed
    raw_speed = target_speed
    
    # Dynamic speed control based on contour tracking and turn severity
    if contour_center is None:
        # Reduce speed when tracking is lost - but not as drastically
        raw_speed = target_speed * 0.7  # Was 0.4, now 0.7
    elif contour_area < 100:
        # Reduce speed when contour is small (less confident)
        raw_speed = target_speed * 0.8  # Was 0.6, now 0.8
    
    # Apply speed caps based on turn sharpness - less aggressive
    turn_severity = abs(new_angle)
    if turn_severity > 0.8:  # Only slow for very sharp turns (was 0.7)
        # Less dramatic slowdown for very sharp turns
        speed_factor = rc_utils.remap_range(turn_severity, 0.6, 1.0, 0.7, 0.5)  # Higher min value
        raw_speed = min(raw_speed, target_speed * speed_factor)
    elif turn_severity > 0.5:  # Was 0.4, now 0.5
        # Less slowdown for medium turns
        speed_factor = rc_utils.remap_range(turn_severity, 0.8, 1, 0.5, 0.7)  # Higher min value
        raw_speed = min(raw_speed, target_speed * speed_factor)
    
    # Never go below minimum speed factor
    raw_speed = max(raw_speed, target_speed * min_speed_factor)
    
    # Faster speed change for more responsive control
    speed_change_rate = 5.0 * delta_time  # Increased from 2.0 for faster response
    if raw_speed > speed:
        speed = min(speed + speed_change_rate, raw_speed)
    else:
        speed = max(speed - speed_change_rate, raw_speed)
    
    # Less smoothing for more responsive steering
    smoothing_factor = min(0.7, 0.3 + (abs(speed) * 0.4))  # Reduced from 0.85 max
    angle = (smoothing_factor * prev_angle) + ((1 - smoothing_factor) * new_angle)
    prev_angle = angle

    # Ensure angle is within bounds
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    if speed < 1:
        speed += 0.1
    speed = rc_utils.clamp(speed, -1.0, 0.5)
    rc.drive.set_max_speed(1)
    speed = rc_utils.clamp(speed, 0.0, 0.8)
    # if angle > 0:
    #     angle += 0.2
    # elif angle < 0:
    #     angle -= 0.2
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    # Set the new speed and angle of the racecar
    print("speed: ", speed)
    print("angle: ", angle)

    rc.drive.set_max_speed(1)
    rc.drive.set_speed_angle(angle, 1)


    # Display status information when holding down certain buttons
    
    # Return the values needed by the caller
    return speed, angle, current_color_index

def update_slow_line():
    """
    Updates the slow information.

    This function checks if a camera image is available, and if so, prints the contour information.
    If no contour is found, it prints a message indicating that no contour was found.
    """
    # Check if a camera image is available
    if rc.camera.get_color_image() is None:
        print("X" * 10 + " (No image) " + "X" * 10)
    else:
        # If no contour is found, print a message indicating that no contour was found
        if contour_center is None:
            print("-" * 32 + " : area = " + str(contour_area))

        # If a contour is found, print the contour information
        else:
            color_name = "Unknown"
            if 0 <= current_color_index < len(COLORS):
                color_name = ["Red", "Green", "Blue"][current_color_index]
            s = ["-"] * 32
            s[int(contour_center[1] / 20)] = "|"
            print("".join(s) + f" : area = {contour_area} ({color_name})")

# Define
def start_line():
    """
    Initializes the racecar's speed and angle.

    This function sets the initial speed and angle of the racecar to 0,
    and then sets the update slow time to 0.5 seconds.
    """
    global speed
    global angle
    global previous_centers
    global prev_angle
    global target_speed
    global recovery_mode
    global recovery_counter
    global last_good_angle
    global last_update_time
    global current_color_index
    global only_show_priority

    speed = 0
    angle = 0
    prev_angle = 0
    previous_centers = []
    target_speed = 1.0  # Start at full speed
    recovery_mode = False
    recovery_counter = 0
    last_good_angle = 0
    last_update_time = 0
    current_color_index = -1
    only_show_priority = False

    # Set the initial speed and angle of the racecar
    rc.drive.set_speed_angle(speed, angle)
    rc.drive.set_max_speed(1.0)  # Allow full speed control

    # Set the update slow time to 0.5 seconds
    rc.set_update_slow_time(0.5)
    


def update_contour_Line():
    """
    Updates the contour information based on the current camera image.

    This function captures the current camera image, crops it to focus on the floor,
    and then searches for contours within the specified color ranges. If a contour
    is found, its center and area are calculated and stored.
    """
    global contour_center
    global contour_area
    global previous_centers
    global recovery_mode
    global recovery_counter
    global last_good_angle
    global current_color_index

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
            hsv_lower = testingColor[0]
            hsv_upper = testingColor[1]
            
            # Create color mask using inRange function
            color_mask = cv.inRange(preprocessed_image, hsv_lower, hsv_upper)
            
            # Filter out shadow areas from the color mask
            filtered_mask = cv.bitwise_and(color_mask, cv.bitwise_not(shadow_mask))
            
            # Find contours in the filtered mask using OpenCV directly
            contours, _ = cv.findContours(filtered_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by minimum area (similar to rc_utils.find_contours)
            valid_contours = [c for c in contours if cv.contourArea(c) > MIN_CONTOUR_AREA]
            
            # If contours are found, check if it's better than the current best
            if valid_contours:
                # Find the largest contour by area
                largest_contour = max(valid_contours, key=cv.contourArea)
                area = cv.contourArea(largest_contour)
                
                # Only consider contours that are not too small
                if area > MIN_CONTOUR_AREA * 2:
                    # Apply a bias toward the currently tracked color to prevent switching
                    if idx == current_color_index:
                        area *= 2.5  # 150% boost to prioritized color (increased from 50%)
                        print(f"Found priority color {idx} with area {area:.1f} (boosted from {area/2.5:.1f})")
                    
                    if area > best_contour_area:
                        best_contour = largest_contour
                        best_contour_area = area
                        best_crop_index = i
                        best_color_index = idx
    
    # Process the best contour if found
    if best_contour is not None and best_contour_area > MIN_CONTOUR_AREA * 3:  # Increased threshold
        # Update the current color being tracked
        current_color_index = best_color_index
        
        # Calculate center using OpenCV moments
        M = cv.moments(best_contour)
        if M["m00"] > 0:  # Prevent division by zero
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            contour_center = (center_y, center_x)  # Note: y, x format to match rc_utils convention
            contour_area = best_contour_area
            
            # Adjust center coordinates based on crop region (needed for consistent steering calculation)
            if best_crop_index > 0:
                # Adjust y coordinate based on the crop offset
                contour_center = (contour_center[0], contour_center[1])
            
            # Store contour centers for predictive steering
            previous_centers.append(contour_center)
            if len(previous_centers) > 4:  # Reduced history for faster response
                previous_centers.pop(0)
            
            # Update last good angle when tracking is reliable
            if contour_area > 100:
                basic_angle = contour_center[1] - rc.camera.get_width() / 2
                basic_angle /= rc.camera.get_width() / 2
                last_good_angle = basic_angle
            
            # Exit recovery mode if we found a good contour
            recovery_mode = False
            recovery_counter = 0
            
            # Draw the contour and its center on the image for the best crop region
            crop_region = crop_regions[best_crop_index]
            cropped_image = rc_utils.crop(image, crop_region[0], crop_region[1])
            
            # Create combined display image
            display_img = cropped_image.copy()
            
            # Draw shadow mask overlay if in debug mode
            if debug_mode and shadow_mask is not None:
                display_img = apply_shadow_overlay_line(display_img, shadow_mask, 0.3)
            
            # Draw a colored border to show the prioritized color
            if current_color_index >= 0:
                # Define border colors for each color index (BGR format)
                border_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
                border_color = border_colors[current_color_index]
                border_thickness = 10
                h, w = display_img.shape[:2]
                # Draw rectangle border
                cv.rectangle(display_img, (0, 0), (w, h), border_color, border_thickness)
            
            # Highlight the color we're tracking
            color_name = ["Red", "Green", "Blue"][best_color_index]
            if current_color_index >= 0:
                priority_color = ["Red", "Green", "Blue"][current_color_index]
                tracking_mode = f"PRIORITY: {priority_color}"
            else:
                tracking_mode = "AUTO"
            cv.putText(display_img, f"Tracking: {color_name} ({tracking_mode})", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw contour and center
            cv.drawContours(display_img, [best_contour], -1, (0, 255, 0), 2)  # Green contour
            cv.circle(display_img, (center_x, center_y), 5, (0, 255, 255), -1)  # Yellow center dot
            
            # Show the image
            rc.display.show_color_image(display_img)
        else:
            # If moments are zero, treat as no contour
            contour_center = None
            contour_area = 0
    else:
        # If no contour is found, enter recovery mode
        contour_center = None
        contour_area = 0
        
        # Keep previous_centers for a short time to help regain tracking
        if len(previous_centers) > 0 and abs(speed) > 0.3:
            # Only clear one point to gradually fade out prediction when tracking is lost
            if len(previous_centers) > 1:
                previous_centers.pop(0)
        else:
            previous_centers = []
        
        # Enter recovery mode if we've lost tracking completely
        if not recovery_mode:
            recovery_mode = True
            recovery_counter = 0
        else:
            # If we're in recovery mode for too long, reset the color tracking
            if recovery_counter > 2.0:  # After 2 seconds of recovery
                current_color_index = -1
            
        # Show the main image with shadow detection
        try:
            main_image = rc_utils.crop(image, crop_regions[0][0], crop_regions[0][1])
            preprocessed = preprocess_Line_image(main_image)
            shadow_mask = create_shadow_mask_Line(preprocessed)
            
            # Create recovery display
            display_img = main_image.copy()
            
            # Add shadow overlay using our safe function
            if debug_mode and shadow_mask is not None:
                display_img = apply_shadow_overlay_line(display_img, shadow_mask, 0.3)
            
            cv.putText(display_img, "SEARCHING... (ignoring shadows)", (10, 30), 
                     cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            rc.display.show_color_image(display_img)
        except Exception as e:
            # If anything fails during recovery display, just show a simple message
            print(f"Error during recovery display: {e}")
            blank_img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv.putText(blank_img, "SEARCHING...", (10, 30), 
                     cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rc.display.show_color_image(blank_img)


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

def preprocess_Line_image(image):
    """
    Preprocesses the image to reduce shadow effects.
    """
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Apply a slight blur to reduce noise
    hsv = cv.GaussianBlur(hsv, (4, 4), 0)
    
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
    print("camera width :", rc.camera.get_width())
    print("camera total height:", rc.camera.get_height())
    if image is not None:
        
        crop_floor = ((400, 160), (rc.camera.get_height(), rc.camera.get_width()))
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
    
    has_largest = isinstance(largestcontour_center, tuple) and len(largestcontour_center) != 2 and largestcontour_center[0] != 0
    has_second = isinstance(secondcontour_center, tuple) and len(secondcontour_center) != 2 and secondcontour_center[0] != 0
    has_general = isinstance(generalcontour_center, tuple) and len(generalcontour_center) != 2 and generalcontour_center[0] != 0
    
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
            speed = 1
        else:
            if center_point < (cameraWidth/2) - 30:
                angle = rc_utils.remap_range(center_point, 0, cameraWidth/2, 0.5, 0.1)
            elif center_point > (cameraWidth/2) + 30:
                angle = rc_utils.remap_range(center_point, cameraWidth/2, cameraWidth, -0.1, -0.5)
            speed = 1
        
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
        
        speed = 1
        
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
    
    speed = 1
    
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
        prioritylist = [ORANGE]
    else:  # Lane_priority == 0
        prioritylist = [PURPLE]
    
    # Get LIDAR scan for wall detection
    scan = rc.lidar.get_samples()
    left_angle, left_distance = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)
    right_angle, right_distance = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    front_angle, front_distance = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    
    # Default values if no point detected
    left_distance = 1000 if left_distance is None else left_distance
    right_distance = 1000 if right_distance is None else right_distance
    front_distance = 1000 if front_distance is None else front_distance
    
    # Follow lanes first
    follow_two_lines_Lane()
    
    # Wall avoidance logic
    wall_avoidance_angle = 0
    
    # If wall is too close on the left, turn right
    if left_distance < 60:
        wall_avoidance_angle = rc_utils.remap_range(left_distance, 20, 60, 0.7, 0.2)
        print(f"Wall on left: {left_distance:.1f}cm, adding angle: {wall_avoidance_angle:.2f}")
    
    # If wall is too close on the right, turn left
    elif right_distance < 60:
        wall_avoidance_angle = rc_utils.remap_range(right_distance, 20, 60, -0.7, -0.2)
        print(f"Wall on right: {right_distance:.1f}cm, adding angle: {wall_avoidance_angle:.2f}")
    
    # If wall is directly in front, make a stronger turn
    if front_distance < 100:
        # Turn away from the closest side wall, or choose left by default if equal
        if left_distance < right_distance:
            wall_avoidance_angle = 0.8  # Turn right
        else:
            wall_avoidance_angle = -0.8  # Turn left
        
        # Slow down when approaching a wall
        speed = rc_utils.remap_range(front_distance, 30, 100, 0.5, 1.0)
        print(f"Wall ahead: {front_distance:.1f}cm, strong avoid: {wall_avoidance_angle:.2f}")
    
    # Combine lane following angle with wall avoidance
    # Lane following gets priority, but wall avoidance can override if needed
    if abs(wall_avoidance_angle) > 0.1:
        # Blend the angles, with more weight to wall avoidance when walls are very close
        blend_factor = rc_utils.remap_range(min(left_distance, right_distance, front_distance), 
                                          20, 60, 0.8, 0.3)
        blend_factor = rc_utils.clamp(blend_factor, 0, 0.8)
        
        # Apply blending
        angle = angle * (1 - blend_factor) + wall_avoidance_angle * blend_factor
    
    # Apply additional steering bias for sharper turns
    if angle > 0:
        angle += 0.2
    elif angle < 0:
        angle -= 0.4
    angle = rc_utils.clamp(angle, -1, 1)
    
    speed_factor = 1.0 - abs(angle) * 1.5
    calculate_speed = speed * max(0.5, speed_factor)
    rc.drive.set_max_speed(0.35)
    calculate_speed = 1
    
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

def update_slalom_contours(image):
    """
    Finds contours for the blue and red cones using color image
    Returns largest contour and its color
    """
    MIN_CONTOUR_AREA = 800

    # If no image is fetched
    if image is None:
        return None, None

    # Find all of the red contours
    contours = rc_utils.find_contours(image, SLALOM_RED[0], SLALOM_RED[1])

    # Find all of the blue contours
    contours_BLUE = rc_utils.find_contours(image, SLALOM_BLUE[0], SLALOM_BLUE[1])

    # Select the largest contour from red and blue contours
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    contour_BLUE = rc_utils.get_largest_contour(contours_BLUE, MIN_CONTOUR_AREA)

    # Process contours
    if contour is not None and contour_BLUE is not None:
        contour_area = rc_utils.get_contour_area(contour)
        contour_area_BLUE = rc_utils.get_contour_area(contour_BLUE)
        
        # If the contour areas are similar enough, indicate that it is a checkpoint
        if abs(contour_area - contour_area_BLUE) < 700:
            return None, "BOTH"
        # If red contour is bigger than the blue one
        elif contour_area > contour_area_BLUE:
            return contour, "RED"
        # If blue contour is equal to or bigger than the red one
        else:
            return contour_BLUE, "BLUE"
    elif contour is None and contour_BLUE is not None:
        return contour_BLUE, "BLUE"
    elif contour_BLUE is None and contour is not None: 
        return contour, "RED"
    else:
        # No contours found
        return None, None

def ID_2_Updtae():
    """
    Handles the cone slaloming behavior when ID 2 is detected
    The car should pass to the right of each red cone and to the left of each blue cone
    """
    global slalom_state, slalom_color_priority, slalom_last_distance, slalom_counter, previous_ID

    # Set previous_ID
    previous_ID = 2
    

    distance = 5000
    dist_param = 200  # Distance parameter for behavior changes

    # Fetch color and depth images
    depth_image = rc.camera.get_depth_image()
    color_image = rc.camera.get_color_image()
    
    if color_image is None or depth_image is None:
        # Can't do anything without images
        rc.drive.stop()
        return

    # Crop the images
    camera_height = (rc.camera.get_height() // 10) * 10
    camera_width = (rc.camera.get_width() // 10) * 10

    tli = (0, rc.camera.get_width() - camera_width)
    bre = ((camera_height, camera_width))
    
    color_image = rc_utils.crop(color_image, tli, bre)
    depth_image = rc_utils.crop(depth_image, tli, bre)

    # Update contours based on color image
    contour, color = update_slalom_contours(color_image)
    
    # Create a copy of the image for display
    image_display = np.copy(color_image)
    
    # Process contours if available
    if contour is not None:
        # Find contour center
        contour_center = rc_utils.get_contour_center(contour)
        
        # Draw contour and center for visualization
        rc_utils.draw_contour(image_display, contour)
        rc_utils.draw_circle(image_display, contour_center)
        
        # Get distance to the contour
        distance = rc_utils.get_pixel_average_distance(depth_image, contour_center)
        slalom_last_distance = distance
    else:
        # No contour found, go to search state
        slalom_state = SlalomState.search

    # Determine state based on detected color
    if color == "RED":
        slalom_state = SlalomState.red
        slalom_color_priority = "RED"
    elif color == "BLUE":
        slalom_state = SlalomState.blue
        slalom_color_priority = "BLUE"
    elif color == "BOTH":
        slalom_state = SlalomState.linear
    
    # Default to moderate speed
    speed = 0.5
    
    # Update car behavior based on current state
    if slalom_state == SlalomState.red:
        # Pass to the right of red cones
        if distance < dist_param:
            # Calculate steering angle based on contour center and distance
            angle = rc_utils.remap_range(contour_center[1], 0, camera_width, 0.3, 1)
            angle *= rc_utils.remap_range(slalom_last_distance, 200, 50, 0, 2)
            slalom_counter = 0
        else:
            # If cone is far, steer based on contour position
            angle = rc_utils.remap_range(contour_center[1], 0, camera_width, -1, 1)
            slalom_counter = 0
    
    elif slalom_state == SlalomState.blue:
        # Pass to the left of blue cones
        if distance < dist_param:
            # Calculate steering angle based on contour center and distance
            angle = rc_utils.remap_range(contour_center[1], 0, camera_width, -1, -0.3)
            angle *= rc_utils.remap_range(slalom_last_distance, 50, 200, 2, 0)
            slalom_counter = 0
        else:
            # If cone is far, steer based on contour position
            angle = rc_utils.remap_range(contour_center[1], 0, camera_width, -1, 1)
            slalom_counter = 0
    
    elif slalom_state == SlalomState.linear:
        # Go straight when both cones are detected similarly
        angle = 0
        slalom_counter = 0
    
    else:  # SlalomState.search
        # No cones detected, continue based on last known color priority
        if slalom_color_priority == "RED":
            angle = rc_utils.remap_range(slalom_last_distance, 1, 100, -0.3, -0.68)
        else:
            angle = rc_utils.remap_range(slalom_last_distance, 1, 100, 0.3, 0.68)
    
    # Enhance steering for responsiveness
    if angle > 1:
        angle += 0.1
    elif angle < 1:
        angle -= 0.1
    
    # Ensure values are within limits
    angle = rc_utils.clamp(angle, -1, 0.8)
    speed = rc_utils.clamp(speed, 0, 1)
    
    # Display the image with contours
    rc.drive.set_max_speed(0.7)
    rc.drive.set_speed_angle(0.8, angle)
    rc.display.show_color_image(image_display)
    return angle
    
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go() 