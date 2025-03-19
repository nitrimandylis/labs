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
Time_start = 4
Time_end = 6
TurnRightValue = 0.7
TrunLeftValue = -0.7
CloseDistance = 70
Distance_To_Start_Alinement = 120

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
            # Don't draw contours here, we'll do it in the DEBUG_MODE section
        
        if contour_blue is not None:
            contour_center_blue = rc_utils.get_contour_center(contour_blue)
            # Don't draw contours here, we'll do it in the DEBUG_MODE section
        
        # Only draw debug UI if enabled
        if DEBUG_MODE:
            # Create a copy of the image for UI rendering
            display_image = image.copy()
            
            # Create a partially transparent overlay for debug info
            debug_overlay = np.zeros_like(display_image)
            
            # Draw colored boxes for detected cones
            if contour_red is not None:
                # Draw red cone with consistent colors (BGR format)
                rc_utils.draw_contour(display_image, contour_red, (0, 0, 255))  # Red
                
                # Draw a direction indicator for red cones (pass on right)
                if contour_center_red is not None:
                    x, y = contour_center_red
                    cv.arrowedLine(display_image, (x, y), (x + 40, y), (0, 0, 255), 2)
                    
                    # Draw selection indicator for RED cone
                    radius = 25
                    thickness = 2
                    # Currently selected cone (solid circle)
                    if cur_state == State.red:
                        cv.circle(display_image, (x, y), radius, (0, 255, 255), thickness)  # Yellow circle
                        cv.putText(display_image, "ACTIVE", (x-25, y-radius-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    # Predicted next cone (dashed circle)
                    elif cur_state == State.blue and Distance_Cone_Red is not None and Distance_Cone_Blue is not None:
                        # Always show next prediction regardless of distance (removed conditional check)
                        # Draw dashed circle
                        for angle in range(0, 360, 30):  # Draw dashed circle segments
                            start_angle = angle
                            end_angle = (angle + 15) % 360
                            start_pt = (x + int(radius * np.cos(np.radians(start_angle))), 
                                        y + int(radius * np.sin(np.radians(start_angle))))
                            end_pt = (x + int(radius * np.cos(np.radians(end_angle))), 
                                    y + int(radius * np.sin(np.radians(end_angle))))
                            cv.line(display_image, start_pt, end_pt, (0, 255, 255), thickness)
                        
                        # Add a second outer circle with a different color to make it more visible
                        outer_radius = radius + 7
                        cv.circle(display_image, (x, y), outer_radius, (0, 255, 0), 1)  # Green outer circle
                        
                        cv.putText(display_image, "NEXT", (x-20, y-radius-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if contour_blue is not None:
                # Draw blue cone with consistent colors (BGR format)
                rc_utils.draw_contour(display_image, contour_blue, (255, 0, 0))  # Blue
                
                # Draw a direction indicator for blue cones (pass on left)
                if contour_center_blue is not None:
                    x, y = contour_center_blue
                    cv.arrowedLine(display_image, (x, y), (x - 40, y), (255, 0, 0), 2)
                    
                    # Draw selection indicator for BLUE cone
                    radius = 25
                    thickness = 2
                    # Currently selected cone (solid circle)
                    if cur_state == State.blue:
                        cv.circle(display_image, (x, y), radius, (0, 255, 255), thickness)  # Yellow circle
                        cv.putText(display_image, "ACTIVE", (x-25, y-radius-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    # Predicted next cone (dashed circle)
                    elif cur_state == State.red and Distance_Cone_Red is not None and Distance_Cone_Blue is not None:
                        # Always show next prediction regardless of distance (removed conditional check)
                        # Draw dashed circle
                        for angle in range(0, 360, 30):  # Draw dashed circle segments
                            start_angle = angle
                            end_angle = (angle + 15) % 360
                            start_pt = (x + int(radius * np.cos(np.radians(start_angle))), 
                                        y + int(radius * np.sin(np.radians(start_angle))))
                            end_pt = (x + int(radius * np.cos(np.radians(end_angle))), 
                                    y + int(radius * np.sin(np.radians(end_angle))))
                            cv.line(display_image, start_pt, end_pt, (0, 255, 255), thickness)
                        
                        # Add a second outer circle with a different color to make it more visible
                        outer_radius = radius + 7
                        cv.circle(display_image, (x, y), outer_radius, (0, 255, 0), 1)  # Green outer circle
                        
                        cv.putText(display_image, "NEXT", (x-20, y-radius-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create a cleaner dashboard UI
            
            # Create a dark semi-transparent panel at the top for key information
            panel_height = 50
            cv.rectangle(debug_overlay, (0, 0), (display_image.shape[1], panel_height), (0, 0, 0), -1)
            cv.addWeighted(debug_overlay, 0.7, display_image, 1, 0, display_image)
            
            # Add main information to the top panel
            font = cv.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)  # White text
            
            # State indicator with color
            state_colors = {
                State.search: (0, 200, 200),      # Yellow for search
                State.red: (0, 0, 255),           # Red for red cone
                State.blue: (255, 0, 0),          # Blue for blue cone
                State.smtng: (150, 150, 150)      # Gray for other
            }
            
            state_color = state_colors.get(cur_state, (255, 255, 255))
            cv.rectangle(display_image, (10, 10), (30, 30), state_color, -1)
            cv.putText(display_image, f"State: {cur_state.name}", (40, 25), font, 0.6, text_color, 1)
            
            # Format distance values
            red_dist = "---"
            if Distance_Cone_Red is not None:  # More reasonable threshold
                red_dist = f"{int(Distance_Cone_Red)}cm"
                
            blue_dist = "---"
            if Distance_Cone_Blue is not None:  # More reasonable threshold
                blue_dist = f"{int(Distance_Cone_Blue)}cm"
            
            # Display distance values in color with better formatting
            cv.putText(display_image, f"Red: {red_dist}", (160, 25), font, 0.6, (0, 0, 255), 1)
            cv.putText(display_image, f"Blue: {blue_dist}", (300, 25), font, 0.6, (255, 0, 0), 1)
            
            # Direction indicator with clear arrow symbol - use ASCII instead of Unicode
            direction_arrow = "^"  # Default forward (up arrow)
            if current_TurnValue is not None:
                if current_TurnValue > 0.1:
                    direction_arrow = ">"  # Right arrow
                elif current_TurnValue < 0:
                    direction_arrow = "<"  # Left arrow
            
            # Show direction symbol in larger font
            cv.putText(display_image, f"{direction_arrow}", (450, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Add secondary information in a small semi-transparent panel
            small_panel_y = panel_height + 5
            small_panel_height = 65
            cv.rectangle(debug_overlay, (5, small_panel_y), (180, small_panel_y + small_panel_height), (0, 0, 0), -1)
            cv.addWeighted(debug_overlay, 0.5, display_image, 1, 0, display_image)
            
            # Add secondary information\
            cv.putText(display_image, f"ProgramTime: {Program_time:.1f}s", (10, small_panel_y + 40), font, 0.5, text_color, 1)
            cv.putText(display_image, f"Timer: {current_time:.1f}s", (10, small_panel_y + 60), font, 0.5, text_color, 1)
            cv.putText(display_image, f"Turn: {angle_error:.2f}", (10, small_panel_y + 80), font, 0.5, text_color, 1)
            
            # Add last turn info
            last_turn_text = "Last turn: ---"
            if Last_Turn is not None:
                last_turn_dir = "RIGHT" if Last_Turn > 0 else "LEFT"
                last_turn_text = f"Last turn: {last_turn_dir}"
            cv.putText(display_image, last_turn_text, (10, small_panel_y + 120), font, 0.5, text_color, 1)
            
            # Mini-map visualization - make it more visible
            map_size = 90
            map_center = (display_image.shape[1] - map_size // 2 - 10, map_size // 2 + 10)
            
            # Draw mini-map background
            cv.circle(display_image, map_center, map_size // 2, (30, 30, 30), -1)  # Dark background
            cv.circle(display_image, map_center, map_size // 2, (100, 100, 100), 2)  # Border
            
            # Draw car indicator
            car_pos = map_center
            car_direction = 90  # Degrees - 0 is right, 90 is up
            
            if current_TurnValue is not None:
                car_direction += current_TurnValue * 45
            
            car_front_x = int(car_pos[0] + np.cos(np.radians(car_direction)) * 15)
            car_front_y = int(car_pos[1] - np.sin(np.radians(car_direction)) * 15)
            
            # Draw car with brighter green for visibility
            cv.circle(display_image, car_pos, 5, (0, 255, 0), -1)
            cv.line(display_image, car_pos, (car_front_x, car_front_y), (0, 255, 0), 2)
            
            # Draw detected cones on mini-map if they exist
            if contour_red is not None and Distance_Cone_Red is not None and Distance_Cone_Red < MAX_DISTANCE:
                # Scale distance to fit on mini-map
                scaled_dist = min(Distance_Cone_Red / 300 * (map_size // 2 - 5), map_size // 2 - 5)
                cone_angle = np.radians(car_direction - angle_error * 30)
                cone_x = int(car_pos[0] + np.cos(cone_angle) * scaled_dist)
                cone_y = int(car_pos[1] - np.sin(cone_angle) * scaled_dist)
                
                # Draw red cone on map with brighter color
                cv.circle(display_image, (cone_x, cone_y), 3, (0, 0, 255), -1)
                
                # Mark selected cone on mini-map
                if cur_state == State.red:
                    cv.circle(display_image, (cone_x, cone_y), 5, (0, 255, 255), 2)
                # Mark as next cone if predicted - always show next indicator
                elif cur_state == State.blue:
                    cv.circle(display_image, (cone_x, cone_y), 7, (0, 255, 0), 1)  # Green circle for next cone
            
            if contour_blue is not None and Distance_Cone_Blue is not None and Distance_Cone_Blue < MAX_DISTANCE:
                # Scale distance to fit on mini-map
                scaled_dist = min(Distance_Cone_Blue / 300 * (map_size // 2 - 5), map_size // 2 - 5)
                cone_angle = np.radians(car_direction - angle_error * 30)
                cone_x = int(car_pos[0] + np.cos(cone_angle) * scaled_dist)
                cone_y = int(car_pos[1] - np.sin(cone_angle) * scaled_dist)
                
                # Draw blue cone on map with brighter color
                cv.circle(display_image, (cone_x, cone_y), 3, (255, 0, 0), -1)
                
                # Mark selected cone on mini-map
                if cur_state == State.blue:
                    cv.circle(display_image, (cone_x, cone_y), 5, (0, 255, 255), 2)
                # Mark as next cone if predicted - always show next indicator
                elif cur_state == State.red:
                    cv.circle(display_image, (cone_x, cone_y), 7, (0, 255, 0), 1)  # Green circle for next cone
            
            # Add a mini legend for the mini-map
            legend_y = map_center[1] + map_size // 2 + 10
            cv.putText(display_image, "Map:", (display_image.shape[1] - 85, legend_y), font, 0.4, (200, 200, 200), 1)
            cv.circle(display_image, (display_image.shape[1] - 55, legend_y), 3, (0, 255, 0), -1)
            cv.putText(display_image, "Car", (display_image.shape[1] - 45, legend_y + 3), font, 0.4, (200, 200, 200), 1)
            cv.circle(display_image, (display_image.shape[1] - 20, legend_y), 3, (0, 0, 255), -1)
            cv.putText(display_image, "R", (display_image.shape[1] - 15, legend_y + 3), font, 0.4, (0, 0, 255), 1)
            cv.circle(display_image, (display_image.shape[1] - 55, legend_y + 15), 3, (255, 0, 0), -1)
            cv.putText(display_image, "B", (display_image.shape[1] - 50, legend_y + 18), font, 0.4, (255, 0, 0), 1)
            
            # Display cursor position in bottom left
            if contour_center is not None:
                x, y = contour_center
                cv.putText(display_image, f"(x={x}, y={y})", (10, display_image.shape[0] - 10), 
                           font, 0.4, (200, 200, 200), 1)
            
            # Display the enhanced image
            rc.display.show_color_image(display_image)

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
    
    # Define TrunLeftValue as global variable now (already defined at the top)
    # TrunLeftValue = -TurnRightValue  # Remove or comment this line
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
        cv.putText(color_image, f"CurrentTurn value: {current_TurnValue:.2f}", (10, 120), font, 0.5, text_color, 1)
        
        # Display last turn direction
        last_turn_text = f"Last turn: {Last_Turn:.2f}" if Last_Turn is not None else "Last turn: None"
        cv.putText(color_image, last_turn_text, (10, 140), font, 0.5, text_color, 1)

        
        rc.display.show_color_image(color_image)
    
    rc.drive.set_speed_angle(0.3, current_TurnValue)

def update_slow():
    pass
    
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()