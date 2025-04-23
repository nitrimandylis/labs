isSimulation = True
import math
import copy
import cv2 as cv
import numpy as np
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
                    print(f"  - ID: {marker.get_id()}, Color: {marker.get_color()}")
    except Exception as e:
        print(f"Error in marker detection: {e}")
def save_current_markers():
    global ID , COLOR
    image = rc.camera.get_color_image()
    if image is not None:
        markers = get_ar_markers(image)
        if markers:
            for marker in markers:
                ID = marker.get_id()
                COLOR = marker.get_color()
def start():
    """Initialize the marker detector"""
    print_info()
    
    # Set update rate for slow update
    rc.set_update_slow_time(0.5)

def update():
    """Main update function for marker detection"""
    current_image = process_detection()

    if current_image is not None:
        rc.display.show_color_image(current_image)

    save_current_markers()
    

def update_slow():
    """Periodic updates for status information"""
    print_current_markers()

# Start the detector
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go() 