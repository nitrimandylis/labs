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

# Global constants
MIN_CONTOUR_AREA = 800
MAX_DISTANCE = 1000000  # More reasonable value for "not found" distance

# Configuration parameters
Time_start = 2
Time_end = 2.7
TurnRightValue = 0.7
CloseDistance = 70
Distance_To_Start_Alinement = 160

# Global variables
cur_state = State.red
current_Cone = Current_Cone.NOTHING
speed = 0.0
angle = 0.0
last_distance = 0
counter = 0
current_time = 0
angle_error = 0
Last_Turn = None
current_TurnValue = None

contour_red = None
contour_blue = None
contour_center_red = None
contour_center_blue = None
contour_center = None
contour_area = 0
Distance_Cone_Red = None
Distance_Cone_Blue = None

# Add at top with other global variables
DEBUG_MODE = True  # Set to False to disable all debug UI for maximum performance
Program_time = 0   # Fixed the missing variable

def update_contours(image, image_depth):
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

    if image is None:
        contour_center = None
        contour_area = 0
        contour_red = None
        contour_blue = None
        contour_center_red = None
        contour_center_blue = None
        Distance_Cone_Red = None
        Distance_Cone_Blue = None
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
            rc_utils.draw_contour(image, contour_red, (255, 0, 0))
        
        if contour_blue is not None:
            contour_center_blue = rc_utils.get_contour_center(contour_blue)
            rc_utils.draw_contour(image, contour_blue, (0, 0, 255))
        
        # Only draw debug UI if enabled
        if DEBUG_MODE:
            # Create a partially transparent overlay for debug info
            debug_overlay = np.zeros_like(image)
            
            # Draw colored boxes for detected cones
            if contour_red is not None:
                rc_utils.draw_contour(image, contour_red, (0, 0, 255))  # BGR format - Red
                
                # Draw a direction indicator for red cones (pass on right)
                if contour_center_red is not None:
                    x, y = contour_center_red
                    cv.arrowedLine(image, (x, y), (x + 40, y), (0, 0, 255), 2)
            
            if contour_blue is not None:
                rc_utils.draw_contour(image, contour_blue, (255, 0, 0))  # BGR format - Blue
                
                # Draw a direction indicator for blue cones (pass on left)
                if contour_center_blue is not None:
                    x, y = contour_center_blue
                    cv.arrowedLine(image, (x, y), (x - 40, y), (255, 0, 0), 2)
            
            # Add semi-transparent black background for text
            cv.rectangle(debug_overlay, (5, 5), (200, 180), (0, 0, 0), -1)
            cv.addWeighted(debug_overlay, 0.5, image, 1, 0, image)
            
            # Add debug text to the image
            font = cv.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)  # White text
            
            # Create visual state indicator
            state_colors = {
                State.search: (200, 200, 0),      # Yellow for search
                State.red: (0, 0, 255),           # Red for red cone
                State.blue: (255, 0, 0),          # Blue for blue cone
                State.smtng: (150, 150, 150)      # Gray for other
            }
            
            # Draw state indicator box
            state_color = state_colors.get(cur_state, (255, 255, 255))
            cv.rectangle(image, (10, 20), (30, 40), state_color, -1)
            
            # Display state with better formatting
            cv.putText(image, f"State: {cur_state.name}", (35, 30), font, 0.5, text_color, 1)
            
            # Display distances with cleaner format
            red_dist = Distance_Cone_Red if Distance_Cone_Red is not None else "---"
            blue_dist = Distance_Cone_Blue if Distance_Cone_Blue is not None else "---"
            
            if isinstance(red_dist, (int, float)):
                red_dist = f"{red_dist:.0f}cm"
            if isinstance(blue_dist, (int, float)):
                blue_dist = f"{blue_dist:.0f}cm"
                
            cv.putText(image, f"Red: {red_dist}", (10, 60), font, 0.5, (0, 0, 255), 1)
            cv.putText(image, f"Blue: {blue_dist}", (10, 80), font, 0.5, (255, 0, 0), 1)
            
            # Display time and control values
            cv.putText(image, f"Timer: {current_time:.1f}s", (10, 100), font, 0.5, text_color, 1)
            cv.putText(image, f"Turn: {angle_error:.2f}", (10, 120), font, 0.5, text_color, 1)
            
            # Show expected turn direction with an arrow
            turn_indicator = "→"
            if current_TurnValue is not None:
                if current_TurnValue > 0.1:
                    turn_indicator = "→"
                elif current_TurnValue < -0.1:
                    turn_indicator = "←"
                else:
                    turn_indicator = "↑"
                    
            cv.putText(image, f"Direction: {turn_indicator}", (10, 140), font, 0.5, text_color, 1)
            
            # Mini-map visualization in corner
            map_size = 80
            map_center = (image.shape[1] - map_size // 2 - 10, map_size // 2 + 10)
            cv.circle(image, map_center, map_size // 2, (50, 50, 50), -1)
            
            # Draw car indicator
            car_pos = map_center
            car_direction = 90  # Degrees - 0 is right, 90 is up
            
            if current_TurnValue is not None:
                car_direction += current_TurnValue * 45
            
            car_front_x = int(car_pos[0] + np.cos(np.radians(car_direction)) * 15)
            car_front_y = int(car_pos[1] - np.sin(np.radians(car_direction)) * 15)
            
            cv.circle(image, car_pos, 5, (0, 255, 0), -1)
            cv.line(image, car_pos, (car_front_x, car_front_y), (0, 255, 0), 2)
            
            # Draw detected cones on mini-map if they exist
            if contour_red is not None and Distance_Cone_Red is not None and Distance_Cone_Red < MAX_DISTANCE:
                # Scale distance to fit on mini-map
                scaled_dist = min(Distance_Cone_Red / 300 * (map_size // 2), map_size // 2)
                cone_angle = np.radians(car_direction - angle_error * 30)
                cone_x = int(car_pos[0] + np.cos(cone_angle) * scaled_dist)
                cone_y = int(car_pos[1] - np.sin(cone_angle) * scaled_dist)
                cv.circle(image, (cone_x, cone_y), 3, (0, 0, 255), -1)
            
            if contour_blue is not None and Distance_Cone_Blue is not None and Distance_Cone_Blue < MAX_DISTANCE:
                # Scale distance to fit on mini-map
                scaled_dist = min(Distance_Cone_Blue / 300 * (map_size // 2), map_size // 2)
                cone_angle = np.radians(car_direction - angle_error * 30)
                cone_x = int(car_pos[0] + np.cos(cone_angle) * scaled_dist)
                cone_y = int(car_pos[1] - np.sin(cone_angle) * scaled_dist)
                cv.circle(image, (cone_x, cone_y), 3, (255, 0, 0), -1)
        
            # Display the enhanced image
            rc.display.show_color_image(image)

        # Continue with the existing logic
        Distance_Cone_Red = MAX_DISTANCE
        Distance_Cone_Blue = MAX_DISTANCE
        
        if image_depth is not None:
            if contour_red is not None and contour_center_red is not None:
                try:
                    depth = image_depth[contour_center_red[0]][contour_center_red[1]]
                    Distance_Cone_Red = depth if depth > 0 else MAX_DISTANCE
                except:
                    Distance_Cone_Red = MAX_DISTANCE
                    
            if contour_blue is not None and contour_center_blue is not None:
                try:
                    depth = image_depth[contour_center_blue[0]][contour_center_blue[1]]
                    Distance_Cone_Blue = depth if depth > 0 else MAX_DISTANCE
                except:
                    Distance_Cone_Blue = MAX_DISTANCE
                            
        if contour_red is not None and contour_blue is None:
            cur_state = State.red
            contour_center = contour_center_red
            cone_Selected = Current_Cone.RED
            contour_area = rc_utils.get_contour_area(contour_red)
            return (cone_Selected, contour_center, contour_area, contour_red)

        if contour_blue is not None and contour_red is None:
            cur_state = State.blue
            contour_center = contour_center_blue
            contour_area = rc_utils.get_contour_area(contour_blue)
            cone_Selected = Current_Cone.BLUE
            return (cone_Selected, contour_center, contour_area, contour_blue)

        if contour_red is not None and contour_blue is not None:                
            if Distance_Cone_Red < Distance_Cone_Blue:
                contour_center = contour_center_red
                contour_area = rc_utils.get_contour_area(contour_red)
                cone_Selected = Current_Cone.RED
                cur_state = State.red
                return (cone_Selected, contour_center, contour_area, contour_red)
            else:
                contour_center = contour_center_blue
                contour_area = rc_utils.get_contour_area(contour_blue)
                cone_Selected = Current_Cone.BLUE
                cur_state = State.blue
                return (cone_Selected, contour_center, contour_area, contour_blue)
        
        return (Current_Cone.NOTHING, None, 0, None)

def start():
    global current_time, Last_Turn, angle_error, current_TurnValue
    global Distance_Cone_Red, Distance_Cone_Blue
    global contour_red, contour_blue, contour_center_red, contour_center_blue
    
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

def update():
    global cur_state, current_Cone, Last_Turn    
    global current_time, angle_error, current_TurnValue
    global Distance_Cone_Blue, Distance_Cone_Red
    global contour_red, contour_blue, contour_center_red, contour_center_blue
    global Program_time
    
    # Update Program_time
    Program_time += rc.get_delta_time()
    
    # Calculate TrunLeftValue based on TurnRightValue
    TrunLeftValue = -TurnRightValue
    current_TurnValue = None
    
    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()
    
    update_contours(color_image, depth_image)
    
    # Handle search state
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
    
    # Update timer
    current_time += rc.get_delta_time()
    if current_time >= Time_end:
        current_time = 0
    
    # Red cone handling
    if Distance_Cone_Red < Distance_Cone_Blue:
        if cur_state == State.red:
            if contour_center_red is not None:
                angle_error = (contour_center_red[1] - 320) / 320
                
                if Distance_Cone_Red < CloseDistance or Distance_Cone_Red > MAX_DISTANCE / 2:
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
    if Distance_Cone_Blue < Distance_Cone_Red:
        if cur_state == State.blue:
            if contour_center_blue is not None:
                angle_error = (contour_center_blue[1] - 320) / 320
                
                if Distance_Cone_Blue < CloseDistance or Distance_Cone_Blue > MAX_DISTANCE / 2:
                    current_TurnValue = TrunLeftValue
                    Last_Turn = TrunLeftValue
                else:
                    if Distance_Cone_Blue > Distance_To_Start_Alinement:
                        current_TurnValue = angle_error - 0.4
                    elif Distance_Cone_Blue < Distance_To_Start_Alinement and Distance_Cone_Blue > CloseDistance:
                        current_TurnValue = angle_error
                    else:
                        current_TurnValue = angle_error
            else:
                cur_state = State.red if contour_red is not None else State.search

    # Counter-turning logic
    if Last_Turn == TurnRightValue:
        if (current_time >= Time_start and 
            Distance_Cone_Blue > Distance_To_Start_Alinement and 
            (CloseDistance > Distance_Cone_Red or Distance_Cone_Red > 1000)):
            current_TurnValue = TrunLeftValue
    elif Last_Turn == TrunLeftValue:
        if (current_time >= Time_start and 
            Distance_Cone_Red > Distance_To_Start_Alinement and 
            (CloseDistance > Distance_Cone_Blue or Distance_Cone_Blue > 1000)):
            current_TurnValue = TurnRightValue
    
    # Set default if no turn value was set
    if current_TurnValue is None:
        current_TurnValue = 0
    
    # Clamp current_TurnValue between -1 and 1
    current_TurnValue = max(-1, min(1, current_TurnValue))
    
    # Remove redundant display code since we're handling it in update_contours
    if color_image is not None and not DEBUG_MODE:
        font = cv.FONT_HERSHEY_COMPLEX
        text_color = (255, 255, 255)  # White text
        
        # Display angle error and turn value
        cv.putText(color_image, f"Angle error: {angle_error:.2f}", (10, 100), font, 0.5, text_color, 1)
        cv.putText(color_image, f"Turn value: {current_TurnValue:.2f}", (10, 120), font, 0.5, text_color, 1)
        
        # Display last turn direction
        last_turn_text = f"Last turn: {Last_Turn:.2f}" if Last_Turn is not None else "Last turn: None"
        cv.putText(color_image, last_turn_text, (10, 140), font, 0.5, text_color, 1)
        
        rc.display.show_color_image(color_image)
    
    rc.drive.set_speed_angle(0.5, current_TurnValue)

def update_slow():
    pass
    
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()