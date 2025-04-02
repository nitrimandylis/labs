"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

"""

########################################################################################
# Imports
########################################################################################

import sys
from enum import Enum
import copy
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
# 34 seconds

class State(Enum):
    RED_CONE_ON_SCREEN = 1,
    RED_CONE_OFF_SCREEN_MOVE_AWAY = 2,
    SEARCHING_AFTER_RED = 3,
    BLUE_CONE_ON_SCREEN = 4,
    BLUE_CONE_OFF_SCREEN_MOVE_AWAY = 5,
    SEARCHING_AFTER_BLUE = 6,
    SEARCHING = 7,
    FOLLOWING_CHECKERED_LINE = 8  # New state for following the checkered line

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

MIN_CONTOUR_AREA = 1000

BLUE = ((90, 100, 50), (120, 255, 255))
RED = ((0, 50, 20), (50, 255, 255))
RED2 = ((160, 50, 20), (179, 255, 255))

# Add more specific color ranges for black and white detection
# Black: Any hue, very low saturation and value
BLACK = ((0, 0, 0), (179, 50, 60))  # Increased upper value limit for black

# White: Any hue, very low saturation, very high value
WHITE = ((0, 0, 180), (179, 40, 255))  # Slightly lower value threshold, higher saturation limit

COLORS = (BLUE, RED, RED2)

GATE_THRESHOLD = 10

color_image = None
contour_center = None
contour_area = None

cur_state = None
timer = None
gate = None
gate_counter = None

# Add any global variables here

########################################################################################
# Functions
########################################################################################

def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center, contour_area, color_image, gate

    gate = False

    image = rc.camera.get_color_image()
    image = image[len(image)//3:]

    if image is None:
        contour_center = None
        contour_area = 0
        contour = None
        color_image = None
    else:
        color_image = copy.deepcopy(image)
        final_contours = []
        contour_centers = []

        for COLOR_RANGE in COLORS:
            # Find all of the blue contours
            contours = rc_utils.find_contours(image, COLOR_RANGE[0], COLOR_RANGE[1])

            # Select the largest contour
            largest_contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
            if largest_contour is not None:
                final_contours.append(largest_contour)
                contour_centers.append(rc_utils.get_contour_center(largest_contour))
        
        if len(contour_centers) == 2:
            gate = True
            contour_center = contour_centers
            contour_area = [rc_utils.get_contour_area(final_contours[0]), 
                          rc_utils.get_contour_area(final_contours[1])]
            for i in range(2):
                rc_utils.draw_contour(image, final_contours[i])
                rc_utils.draw_circle(image, contour_centers[i])
        else:
            # Get the largest contour for single cone case
            largest_contour = rc_utils.get_largest_contour(final_contours, MIN_CONTOUR_AREA)
            if largest_contour is not None:
                contour_center = rc_utils.get_contour_center(largest_contour)
                contour_area = rc_utils.get_contour_area(largest_contour)
                rc_utils.draw_contour(image, largest_contour)
                rc_utils.draw_circle(image, contour_center)
            else:
                contour_center = None
                contour_area = 0

        # No need to always display this - we're showing the line detection debug image
        # rc.display.show_color_image(image)
    

def is_blue(bgr_value):
    return 100 < bgr_value[0]


def detect_checkered_line():
    """
    Checkerboard pattern detector specifically optimized for the simulation environment.
    Focuses on exact black-white pattern with perfect spacing.
    """
    image = rc.camera.get_color_image()
    if image is None:
        return False, 0
    
    # SPECIFICALLY target the area where the checkered flag appears in the simulation
    # This is more targeted than previous attempts
    height = len(image)
    width = len(image[0])
    
    # The flag is in the middle of the view, at a specific height
    # Focus on exactly where it should be based on the image you provided
    top = int(height * 0.45)     # Start higher up - flag is visible in middle of screen
    bottom = int(height * 0.55)  # Just a narrow band where the checkered pattern is
    
    # Create a debug image for the original cropped region
    cropped = image[top:bottom, :]
    debug_original = copy.deepcopy(cropped)
    
    # Convert to grayscale with high contrast
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    
    # Apply aggressive binary thresholding - specifically for simulation lighting
    # This is much stronger than previous attempts
    _, binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
    
    # Create debug image of processed binary
    debug_binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    
    # Sample the middle row (where the checkered pattern is most visible)
    mid_row = binary.shape[0] // 2
    row = binary[mid_row, :]
    
    # Highlight this row to show where we're checking
    debug_binary[mid_row, :] = [0, 255, 255]  # Yellow line
    
    # Count transitions from black to white and white to black
    transitions = []
    last_val = row[0]
    for i in range(1, len(row)):
        # Only consider strong black-white transitions
        if (row[i] == 0 and last_val == 255) or (row[i] == 255 and last_val == 0):
            transitions.append(i)
            last_val = row[i]
            # Mark on the debug image
            cv.circle(debug_binary, (i, mid_row), 2, (0, 0, 255), -1)  # Red dot
    
    # Need at least 6 transitions for a checkerboard (3 squares)
    if len(transitions) >= 6:
        # Calculate distances between transitions
        distances = [transitions[i] - transitions[i-1] for i in range(1, len(transitions))]
        
        # Calculate average distance and standard deviation
        if len(distances) >= 5:
            avg_dist = sum(distances) / len(distances)
            std_dev = np.std(distances)
            
            # Check for extremely consistent spacing - key feature of checkerboard
            # This is much stricter than previous attempts
            if std_dev / avg_dist < 0.2 and 15 < avg_dist < 40:
                
                # CRITICAL CHECK: Verify first transition starts near image center
                # The checkered flag in the simulation is centered
                if abs(transitions[0] - (width // 2)) < width // 4:
                    
                    # Verify last transition isn't at the image edge (which could be a shadow)
                    if abs(transitions[-1] - width) > 20:
                        
                        # Calculate the center of the pattern
                        center_x = sum(transitions) / len(transitions)
                        
                        # Create a composite debug image
                        # Top: Original cropped color image
                        # Bottom: Binary with detection markers
                        debug_height = debug_original.shape[0] + debug_binary.shape[0]
                        debug_img = np.zeros((debug_height, width, 3), dtype=np.uint8)
                        debug_img[:debug_original.shape[0], :] = debug_original
                        debug_img[debug_original.shape[0]:, :] = debug_binary
                        
                        # Draw center line on both parts of the debug image
                        center_x_int = int(center_x)
                        # Line on original
                        cv.line(debug_img, (center_x_int, 0), 
                                (center_x_int, debug_original.shape[0]), 
                                (0, 255, 0), 2)
                        # Line on binary
                        cv.line(debug_img, (center_x_int, debug_original.shape[0]), 
                                (center_x_int, debug_height), 
                                (0, 255, 0), 2)
                        
                        # Add text to show key metrics for debugging
                        text = f"Transitions: {len(transitions)}, Avg Dist: {avg_dist:.1f}, StdDev: {std_dev:.1f}"
                        cv.putText(debug_img, text, (10, 20),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Show the debug image
                        rc.display.show_color_image(debug_img)
                        
                        return True, center_x
    
    # No valid pattern found, still show debug composite
    if 'debug_original' in locals() and 'debug_binary' in locals():
        debug_height = debug_original.shape[0] + debug_binary.shape[0]
        debug_img = np.zeros((debug_height, width, 3), dtype=np.uint8)
        debug_img[:debug_original.shape[0], :] = debug_original
        debug_img[debug_original.shape[0]:, :] = debug_binary
        rc.display.show_color_image(debug_img)
    
    return False, 0


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    global timer, cur_state, color_image, contour_center, contour_area, gate, gate_counter
    timer = 0.0
    cur_state = State.SEARCHING
    color_image = None
    contour_center = None
    contour_area = None
    gate = False
    gate_counter = 0



def update_timer():
    global timer
    timer += rc.get_delta_time()
    


def get_gate_angle() -> float:
    kpa = 0.001
    angle1 = kpa * (contour_center[0][1] - (rc.camera.get_width() / 2))
    angle2 = kpa * (contour_center[1][1] - (rc.camera.get_width() / 2))
    return min(1.0, max(-1.0, (angle1 + angle2)/2))


def update():
    global timer, cur_state, color_image, contour_center, contour_area, gate_counter, gate , ProgramTime
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    update_contour()
    update_timer()

    # Check for checkered flag pattern
    line_detected, line_center = detect_checkered_line()
    
    # Debug output
    print(f"Checkered flag detected: {line_detected}, Center: {line_center}")
    
    # Static counter for consecutive detections
    static_line_counter = getattr(update, 'static_line_counter', 0)
    
    if line_detected:
        static_line_counter += 1
        # Require at least 2 consecutive detections to filter out false positives
        if static_line_counter >= 2:
            print("CHECKERED FLAG CONFIRMED! Switching to finish line following.")
            cur_state = State.FOLLOWING_CHECKERED_LINE
            timer = 0.0  # Reset timer
    else:
        static_line_counter = 0
    
    # Store counter as function attribute
    update.static_line_counter = static_line_counter
    ProgramTime = rc.get_delta_time()
    print(ProgramTime)
    # Process the current state
    if color_image is not None and ProgramTime < 34:
        speed = 1.0
        angle = None

        # Handle checkered line following (highest priority)
        if cur_state == State.FOLLOWING_CHECKERED_LINE:
            print('FOLLOWING CHECKERED LINE')
            if line_detected:
                # Calculate steering angle based on line position
                image_width = rc.camera.get_width()
                
                # Center the car on the line
                normalized_pos = (line_center - (image_width/2)) / (image_width/2)
                
                # Very gentle steering for finish line
                angle = normalized_pos * 0.4
                angle = max(-1, min(1, angle))
                
                # Maintain moderate speed
                speed = 0.6
                timer = 0.0  # Reset timer when line is detected
            else:
                # If temporarily lose the line, maintain straight path
                print("Line lost, continuing straight")
                speed = 0.4
                angle = 0  # Go straight
                
                # After a longer delay, revert to cone navigation
                if timer > 1.0:  # One full second
                    print("Pattern lost for 1 second, returning to cone navigation")
                    cur_state = State.SEARCHING
                    timer = 0.0
                    static_line_counter = 0  # Reset detection counter
        
        # Original cone slalom logic
        elif contour_center is not None:
            kpa = 0.2

            depth_image = rc.camera.get_depth_image()
            depth_image = depth_image[len(depth_image)//3:]

            if gate:
                # For gate, contour_center is a list of two points
                distance = depth_image[contour_center[0][0]][contour_center[0][1]]
                color = color_image[contour_center[0][0]][contour_center[0][1]]
                angle_to_cone = get_gate_angle()
                gate_counter += 1
            else:
                # For single contour, contour_center is a single point
                distance = depth_image[contour_center[0]][contour_center[1]]
                color = color_image[contour_center[0]][contour_center[1]]
                if gate_counter < GATE_THRESHOLD:
                    angle_to_cone = max(-1, min(kpa * (contour_center[1] - (rc.camera.get_width() / 2)), 1))
                elif is_blue(color):
                    angle_to_cone = -0.3
                else:
                    angle_to_cone = 0.3

            gate = gate or gate_counter >= GATE_THRESHOLD

            blue = False
            red = False

            if is_blue(color):
                blue = True
                if (cur_state == State.SEARCHING_AFTER_RED) or (cur_state == State.SEARCHING and distance < 120):
                    cur_state = State.BLUE_CONE_ON_SCREEN
                    timer = 0.0
            else:
                red = True
                if (cur_state == State.SEARCHING_AFTER_BLUE) or (cur_state == State.SEARCHING and distance < 120):
                    cur_state = State.RED_CONE_ON_SCREEN
                    timer = 0.0

            if cur_state == State.BLUE_CONE_ON_SCREEN:
                angle = -1
            
            if cur_state == State.RED_CONE_ON_SCREEN:
                angle = 1
            
            if (gate or distance > 90) and ((cur_state == State.BLUE_CONE_ON_SCREEN and blue) or (cur_state == State.RED_CONE_ON_SCREEN and red)):
                angle = angle_to_cone

        if contour_center is None:
            if cur_state == State.RED_CONE_ON_SCREEN and gate_counter < GATE_THRESHOLD:
                gate_counter = 0
                cur_state = State.RED_CONE_OFF_SCREEN_MOVE_AWAY
                timer = 0.0
            elif cur_state == State.RED_CONE_ON_SCREEN:
                cur_state = State.SEARCHING
            
            if cur_state == State.BLUE_CONE_ON_SCREEN and gate_counter < GATE_THRESHOLD:
                gate_counter = 0
                cur_state = State.BLUE_CONE_OFF_SCREEN_MOVE_AWAY
                timer = 0.0
            elif cur_state == State.BLUE_CONE_ON_SCREEN:
                cur_state = State.SEARCHING
        
        if cur_state == State.RED_CONE_OFF_SCREEN_MOVE_AWAY:
            if timer > 0.35:
                cur_state = State.SEARCHING_AFTER_RED
                timer = 0.0
            else:
                angle = 1
        
        if cur_state == State.BLUE_CONE_OFF_SCREEN_MOVE_AWAY:
            if timer > 0.35:
                cur_state = State.SEARCHING_AFTER_BLUE
                timer = 0.0
            else:
                angle = -0.7
        
        if cur_state == State.SEARCHING:
            angle = 0
        
        if cur_state == State.SEARCHING_AFTER_RED:
            angle = -1
        
        if cur_state == State.SEARCHING_AFTER_BLUE:
            angle = 1
        
        print('contour centers: ' + str(contour_center))
        print('isGate: ' + str(gate))
        print('gate counter: ' + str(gate_counter))
        print('angle: ' + str(angle))
        print(cur_state)
        rc.drive.set_speed_angle(speed, angle)
    else:
        rc.drive.set_speed_angle(1, -1)
        if ProgramTime > 35:
            rc.drive.stop()


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()