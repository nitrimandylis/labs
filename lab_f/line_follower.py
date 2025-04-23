#Just do line following  with purple and orang . set it to read the color from AR marker (for now just add an option to set color by pressing x for oranga and y for purple) to set either purple or orange as the primery coliur then make the seccondary color make the car go a tiny bit solwer or dont it might still work

import sys
import cv2 as cv
import numpy as np
import math

# Import the racecar_core and racecar_utils libraries
sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

# Create a racecar object
rc = racecar_core.create_racecar()

# Define constants
MIN_CONTOUR_AREA = 30
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# Define color ranges for purple and orange
COLORS = [
    # Purple color range
    ((125, 100, 100), (140, 255, 255)),
    # Orange color range
    ((10, 150, 120), (25, 255, 255))
]

# Shadow detection constants
SHADOW_MAX_VALUE = 80
SHADOW_MAX_SATURATION = 60

# Initialize variables
speed = 0.0
angle = 0.0
contour_center = None
contour_area = 0

# Tracking variables
previous_centers = []
prev_angle = 0
target_speed = 0.8
last_update_time = 0
recovery_mode = False
recovery_counter = 0
last_good_angle = 0
min_speed_factor = 0.7
current_color_index = -1  # -1: auto, 0: purple, 1: orange
debug_mode = True
only_show_priority = False

# Add this to your globals near the top of the file:
line_offset = 0.0  # Positive = right of line, negative = left of line
line_offset_step = 0.05  # How much to change offset with each button press

# Add these variables to the global variables section
left_contour_center = None
right_contour_center = None
left_contour_area = 0
right_contour_area = 0
dual_line_mode = False  # Toggle with controller button

# Add these variables to the global section
current_line_side = "unknown"  # "left", "right", or "unknown"
image_center_x = None  # Will be initialized when first image is processed

# Add this to the global section
last_detected_line_side = "unknown"  # Stores the last definitive line side (left or right)

def preprocess_image(image):
    """Preprocesses the image to reduce shadow effects."""
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Apply a slight blur to reduce noise
    hsv = cv.GaussianBlur(hsv, (5, 5), 0)
    
    # Apply CLAHE on the V channel to normalize brightness
    h, s, v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Enhance saturation
    s = cv.multiply(s, 1.3)
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # Merge channels back
    hsv = cv.merge([h, s, v])
    
    return hsv

def create_shadow_mask(hsv_image):
    """Creates a binary mask that identifies shadow areas."""
    # Split HSV channels
    h, s, v = cv.split(hsv_image)
    
    # Create shadow mask where both saturation and value are low
    shadow_mask = cv.bitwise_and(
        cv.threshold(s, SHADOW_MAX_SATURATION, 255, cv.THRESH_BINARY_INV)[1],
        cv.threshold(v, SHADOW_MAX_VALUE, 255, cv.THRESH_BINARY_INV)[1]
    )
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv.morphologyEx(shadow_mask, cv.MORPH_OPEN, kernel)
    shadow_mask = cv.morphologyEx(shadow_mask, cv.MORPH_CLOSE, kernel)
    
    return shadow_mask

def apply_shadow_overlay(image, shadow_mask, opacity=0.3):
    """Safely applies a shadow overlay to an image."""
    try:
        # Make sure shadow mask has the same dimensions
        if shadow_mask.shape[:2] != image.shape[:2]:
            shadow_mask = cv.resize(shadow_mask, (image.shape[1], image.shape[0]))
        
        # Convert mask to BGR for overlay
        shadow_overlay = cv.cvtColor(shadow_mask, cv.COLOR_GRAY2BGR)
        shadow_overlay[:, :, 0] = 0  # Set B channel to 0
        shadow_overlay[:, :, 1] = 0  # Set G channel to 0
        # Keep R channel to show shadows as red
        
        # Blend the images
        return cv.addWeighted(image, 1.0, shadow_overlay, opacity, 0)
    except Exception as e:
        print(f"Warning: Failed to apply shadow overlay: {e}")
        return image

def determine_line_side(contour_center):
    """
    Determines if the car is following the right or left line based on contour position.
    Returns "left", "right", or "unknown"
    """
    global image_center_x
    global last_detected_line_side
    
    if contour_center is None or image_center_x is None:
        return "unknown"
    
    # Determine which side of the image the contour is on
    contour_x = contour_center[1]
    
    # Consider the contour to be on the right if it's in the right 40% of the image
    # and on the left if it's in the left 40% of the image
    if contour_x > image_center_x * 1.2:  # Right 40% of image
        last_detected_line_side = "right"
        return "right"
    elif contour_x < image_center_x * 0.8:  # Left 40% of image
        last_detected_line_side = "left"
        return "left"
    else:
        # In the middle - return the last detected side instead of "center"
        return last_detected_line_side if last_detected_line_side != "unknown" else "unknown"

def update_contour():
    """Updates the contour information based on the current camera image."""
    global contour_center
    global contour_area
    global previous_centers
    global recovery_mode
    global recovery_counter
    global last_good_angle
    global current_color_index
    global line_offset
    global dual_line_mode
    global left_contour_center
    global right_contour_center
    global left_contour_area
    global right_contour_area
    global current_line_side
    global image_center_x

    # Capture the current camera image
    image = rc.camera.get_color_image()

    # If the image is None, reset all contour information
    if image is None:
        contour_center = None
        contour_area = 0
        left_contour_center = None
        right_contour_center = None
        left_contour_area = 0
        right_contour_area = 0
        return
    
    # Initialize the image center if not already set
    if image_center_x is None:
        image_center_x = rc.camera.get_width() / 2
    
    # Use multiple crop regions for more robust tracking
    crop_regions = []
    
    # Main crop region - dynamic based on speed
    if abs(speed) > 0.7:
        crop_y = 300  # Look further ahead at higher speeds (was 300)
    else:
        crop_y = 340  # Default crop height (was 340)
    
    # Add main crop region
    crop_regions.append(((crop_y, 0), (rc.camera.get_height(), rc.camera.get_width())))
    
    # If in recovery mode or at high speed, add a wider/lower crop region
    if recovery_mode or abs(speed) > 0.6:
        crop_regions.append(((380, 0), (rc.camera.get_height(), rc.camera.get_width())))  # Was 380
    
    # Now add a third, even higher crop region for better visibility at high speeds
    if abs(speed) > 0.8:
        crop_regions.append(((200, 0), (rc.camera.get_height(), rc.camera.get_width())))
        
    # Initialize variables for contour detection
    best_contour = None
    best_contour_area = 0
    best_crop_index = 0
    best_color_index = -1
    shadow_mask = None
    
    # Variables for dual line tracking
    all_valid_contours = []
    
    # Try each crop region
    for i, crop_region in enumerate(crop_regions):
        # Crop the image with the current crop region
        cropped_image = rc_utils.crop(image, crop_region[0], crop_region[1])
        
        # Preprocess the image to reduce shadow effects
        preprocessed_image = preprocess_image(cropped_image)
        
        # Create shadow mask
        shadow_mask = create_shadow_mask(preprocessed_image)
        
        # Show shadow mask in debug mode
        if debug_mode and i == 0:
            debug_img = cropped_image.copy()
            debug_img = apply_shadow_overlay(debug_img, shadow_mask, 0.5)
            cv.putText(debug_img, "Red = Shadow Areas", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rc.display.show_color_image(debug_img)
        
        # Process color ranges to find contours
        color_indices = list(range(len(COLORS)))
        if current_color_index >= 0:
            if only_show_priority:
                color_indices = [current_color_index]
            else:
                color_indices.remove(current_color_index)
                color_indices.insert(0, current_color_index)
                
        # Print what we're checking in update_contour
        if i == 0 and (current_color_index >= 0 or recovery_mode):
            if current_color_index >= 0:
                mode_str = "EXCLUSIVE" if only_show_priority else "PRIORITY"
                print(f"Checking colors in {mode_str} mode: {[['Purple', 'Orange'][i] for i in color_indices]}")
            elif recovery_mode:
                print(f"In recovery mode... searching all colors")

        # Process each color to find contours
        for idx in color_indices:
            testingColor = COLORS[idx]
            hsv_lower = testingColor[0]
            hsv_upper = testingColor[1]
            
            # Create color mask using inRange function
            color_mask = cv.inRange(preprocessed_image, hsv_lower, hsv_upper)
            
            # Filter out shadow areas from the color mask
            filtered_mask = cv.bitwise_and(color_mask, cv.bitwise_not(shadow_mask))
            
            # Find contours in the filtered mask
            contours, _ = cv.findContours(filtered_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by minimum area
            valid_contours = [c for c in contours if cv.contourArea(c) > MIN_CONTOUR_AREA]
            
            # In dual line mode, store all valid contours for later processing
            if dual_line_mode and valid_contours:
                for contour in valid_contours:
                    all_valid_contours.append({
                        'contour': contour,
                        'area': cv.contourArea(contour),
                        'crop_index': i,
                        'color_index': idx
                    })
            
            # For single line mode or as fallback, find the best contour as before
            if valid_contours:
                largest_contour = max(valid_contours, key=cv.contourArea)
                area = cv.contourArea(largest_contour)
                
                if area > MIN_CONTOUR_AREA * 2:
                    if idx == current_color_index:
                        area *= 2.5  # Boost to prioritized color
                    
                    if area > best_contour_area:
                        best_contour = largest_contour
                        best_contour_area = area
                        best_crop_index = i
                        best_color_index = idx
    
    # Process contours based on mode
    if dual_line_mode and len(all_valid_contours) >= 2:
        # Process dual line mode - sort contours by x position to determine left vs right
        valid_contours_with_centers = []
        
        # Calculate centers for all contours
        for contour_data in all_valid_contours:
            contour = contour_data['contour']
            M = cv.moments(contour)
            if M["m00"] > 0:  # Prevent division by zero
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                valid_contours_with_centers.append({
                    'contour': contour,
                    'center': (center_y, center_x),
                    'area': contour_data['area'],
                    'crop_index': contour_data['crop_index'],
                    'color_index': contour_data['color_index'],
                    'x_pos': center_x  # Use this to sort left/right
                })
        
        # Sort by x position (left to right)
        if valid_contours_with_centers:
            valid_contours_with_centers.sort(key=lambda c: c['x_pos'])
            
            # Get the leftmost and rightmost contours 
            if len(valid_contours_with_centers) >= 2:
                # Left contour is the leftmost
                left_data = valid_contours_with_centers[0]
                left_contour_center = left_data['center']
                left_contour_area = left_data['area']
                
                # Right contour is the rightmost
                right_data = valid_contours_with_centers[-1]
                right_contour_center = right_data['center']
                right_contour_area = right_data['area']
                
                # For the main contour (for display/fallback), use the one with larger area
                if left_contour_area > right_contour_area:
                    contour_center = left_contour_center
                    contour_area = left_contour_area
                    best_contour = left_data['contour']
                    best_crop_index = left_data['crop_index']
                    best_color_index = left_data['color_index']
                else:
                    contour_center = right_contour_center
                    contour_area = right_contour_area
                    best_contour = right_data['contour']
                    best_crop_index = right_data['crop_index']
                    best_color_index = right_data['color_index']
                
                # Update other tracking info
                recovery_mode = False
                recovery_counter = 0
                
                # Calculate a center point between the lines for display purposes
                center_x = (left_contour_center[1] + right_contour_center[1]) // 2
                center_y = (left_contour_center[0] + right_contour_center[0]) // 2
                mid_center = (center_y, center_x)
                
                # Store for predictive steering
                previous_centers.append(mid_center)
                if len(previous_centers) > 4:
                    previous_centers.pop(0)
                
                # Update last good angle - use the center between the lines
                basic_angle = mid_center[1] - rc.camera.get_width() / 2
                basic_angle /= rc.camera.get_width() / 2
                last_good_angle = basic_angle
            else:
                # Only one contour found, use single contour mode
                left_contour_center = None
                right_contour_center = None
                left_contour_area = 0
                right_contour_area = 0
                
                # Fall back to single contour mode below
    
    # Process the best contour if found (for single contour mode or as fallback)
    if not dual_line_mode or left_contour_center is None or right_contour_center is None:
        # Process single contour as in original code
        if best_contour is not None and best_contour_area > MIN_CONTOUR_AREA * 3:
            # Update the current color being tracked
            current_color_index = best_color_index
            
            # Calculate center using OpenCV moments
            M = cv.moments(best_contour)
            if M["m00"] > 0:  # Prevent division by zero
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                contour_center = (center_y, center_x)  # y, x format to match rc_utils convention
                contour_area = best_contour_area
                
                # Store contour centers for predictive steering
                previous_centers.append(contour_center)
                if len(previous_centers) > 4:  # Keep only recent history
                    previous_centers.pop(0)
                
                # Update last good angle when tracking is reliable
                if contour_area > 100:
                    basic_angle = contour_center[1] - rc.camera.get_width() / 2
                    basic_angle /= rc.camera.get_width() / 2
                    last_good_angle = basic_angle
                
                # Exit recovery mode if we found a good contour
                recovery_mode = False
                recovery_counter = 0
            else:
                # If moments are zero, treat as no contour
                contour_center = None
                contour_area = 0
        else:
            # If no contour is found, enter recovery mode
            contour_center = None
            contour_area = 0
            left_contour_center = None
            right_contour_center = None
            left_contour_area = 0
            right_contour_area = 0
            
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
                preprocessed = preprocess_image(main_image)
                shadow_mask = create_shadow_mask(preprocessed)
                
                # Create recovery display
                display_img = main_image.copy()
                
                # Add shadow overlay
                if debug_mode and shadow_mask is not None:
                    display_img = apply_shadow_overlay(display_img, shadow_mask, 0.3)
                
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

    # After processing contours and finding the best contour
    if not dual_line_mode and contour_center is not None:
        # Determine which line we're following
        current_line_side = determine_line_side(contour_center)
    
    # Display logic with modifications for dual line mode
    if (dual_line_mode and left_contour_center is not None and right_contour_center is not None) or \
       (not dual_line_mode and contour_center is not None):
        # Get the crop region for display
        crop_region = crop_regions[best_crop_index]
        cropped_image = rc_utils.crop(image, crop_region[0], crop_region[1])
        
        # Create combined display image
        display_img = cropped_image.copy()
        
        # Draw shadow mask overlay if in debug mode
        if debug_mode and shadow_mask is not None:
            display_img = apply_shadow_overlay(display_img, shadow_mask, 0.3)
        
        # Draw a colored border to show the prioritized color
        if current_color_index >= 0:
            # Define border colors for each color index (BGR format)
            border_colors = [(255, 0, 255), (0, 165, 255)]  # Purple, Orange
            border_color = border_colors[current_color_index]
            border_thickness = 10
            h, w = display_img.shape[:2]
            # Draw rectangle border
            cv.rectangle(display_img, (0, 0), (w, h), border_color, border_thickness)
        
        # Highlight the mode we're in
        if dual_line_mode:
            mode_str = "DUAL LINE MODE"
            
            # Draw left line contour and marker (red)
            left_center_x = left_contour_center[1]
            left_center_y = left_contour_center[0]
            cv.circle(display_img, (left_center_x, left_center_y), 5, (0, 0, 255), -1)  # Red dot
            
            # Draw right line contour and marker (blue)
            right_center_x = right_contour_center[1]
            right_center_y = right_contour_center[0]
            cv.circle(display_img, (right_center_x, right_center_y), 5, (255, 0, 0), -1)  # Blue dot
            
            # Draw middle point between lines (yellow)
            mid_x = (left_center_x + right_center_x) // 2
            mid_y = (left_center_y + right_center_y) // 2
            cv.circle(display_img, (mid_x, mid_y), 8, (0, 255, 255), -1)  # Yellow dot
            
            # Draw line connecting the two lines
            cv.line(display_img, (left_center_x, left_center_y), 
                   (right_center_x, right_center_y), (0, 255, 0), 2)
            
            # Add text showing line distances
            lane_width = abs(right_center_x - left_center_x)
            cv.putText(display_img, f"Lane width: {lane_width}px", (10, 90),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Single line mode - draw contour and center as before
            color_name = ["Purple", "Orange"][best_color_index]
            if current_color_index >= 0:
                priority_color = ["Purple", "Orange"][current_color_index]
                mode_str = f"SINGLE LINE: {priority_color}"
            else:
                mode_str = "SINGLE LINE: AUTO"
            
            # Draw contour and center
            cv.drawContours(display_img, [best_contour], -1, (0, 255, 0), 2)  # Green contour
            cv.circle(display_img, (contour_center[1], contour_center[0]), 5, (0, 255, 255), -1)  # Yellow center dot
        
        # Display mode text
        cv.putText(display_img, mode_str, (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add line offset indicator if applicable
        if not dual_line_mode and debug_mode:
            target_x_px = int(contour_center[1] + (line_offset * rc.camera.get_width() / 2))
            cv.line(display_img, 
                    (target_x_px, contour_center[0] - 20), 
                    (target_x_px, contour_center[0] + 20), 
                    (0, 255, 255), 2)  # Yellow vertical line
            cv.putText(display_img, f"Offset: {line_offset:.2f}", (10, 50),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add line side indicator if not in dual line mode
        if not dual_line_mode:
            # Add a text indicator showing which line we're following
            line_side_color = (0, 255, 255)  # Yellow for center
            if current_line_side == "right":
                line_side_color = (255, 0, 0)  # Blue for right line
            elif current_line_side == "left":
                line_side_color = (0, 0, 255)  # Red for left line
            
            cv.putText(display_img, f"Following: {current_line_side.upper()} line", 
                      (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, line_side_color, 2)
            
            # Draw a vertical line at the image center for reference
            cv.line(display_img, 
                    (int(image_center_x), 0), 
                    (int(image_center_x), display_img.shape[0]), 
                    (255, 255, 255), 1)  # White center line
        
        # Show the image
        rc.display.show_color_image(display_img)

def start():
    """Initializes the racecar's speed and angle."""
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
    global line_offset
    global line_offset_step
    global dual_line_mode
    global left_contour_center
    global right_contour_center
    global left_contour_area
    global right_contour_area
    global current_line_side
    global image_center_x
    global last_detected_line_side

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
    line_offset = 0
    line_offset_step = 0.05
    dual_line_mode = False
    left_contour_center = None
    right_contour_center = None
    left_contour_area = 0
    right_contour_area = 0
    current_line_side = "unknown"
    last_detected_line_side = "unknown"
    image_center_x = None  # Will be set when first image is processed

    # Set the initial speed and angle of the racecar
    rc.drive.set_speed_angle(speed, angle)
    rc.drive.set_max_speed(1.0)  # Allow full speed control

    # Set the update slow time
    rc.set_update_slow_time(0.5)
    
    print("Line follower initialized with PURPLE and ORANGE detection")
    print("CONTROLS:")
    print("X: Set ORANGE as primary color")
    print("Y: Set PURPLE as primary color")
    print("B: Reset color priority (track both colors)")
    print("A: Toggle dual line mode (track left and right lines)")
    print("LB/RB: Adjust line offset")
    print("Line side detection: Will display which line (LEFT/RIGHT) is being followed")

def update():
    """Updates the racecar's speed and angle based on the current contour information."""
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
    global line_offset
    global dual_line_mode

    # Calculate delta time for smoother motion and timing
    current_time = rc.get_delta_time()
    delta_time = current_time - last_update_time
    last_update_time = current_time

    # Update the contour information
    update_contour()

    # Add dual line mode toggle with A button
    if rc.controller.was_pressed(rc.controller.Button.A):
        dual_line_mode = not dual_line_mode
        print(f"Dual line mode: {'ENABLED' if dual_line_mode else 'DISABLED'}")

    # Add these controller checks for line offset adjustment
    if rc.controller.was_pressed(rc.controller.Button.LB):
        # Move more to the left of the line
        line_offset -= line_offset_step
        print(f"Line offset: {line_offset:.2f} (left of line)")
    
    if rc.controller.was_pressed(rc.controller.Button.RB):
        # Move more to the right of the line
        line_offset += line_offset_step
        print(f"Line offset: {line_offset:.2f} (right of line)")

    # Color selection with controller buttons
    if rc.controller.was_pressed(rc.controller.Button.X):
        current_color_index = 1  # Set Orange as primary
        only_show_priority = True
        print("Setting ORANGE as primary color")
    
    if rc.controller.was_pressed(rc.controller.Button.Y):
        current_color_index = 0  # Set Purple as primary
        only_show_priority = True
        print("Setting PURPLE as primary color")
        
    if rc.controller.was_pressed(rc.controller.Button.B):
        current_color_index = -1  # Reset to track both colors
        only_show_priority = False
        print("Resetting color priority - will track both colors")

    new_angle = 0
    
    # Modified steering logic to handle dual line mode
    if dual_line_mode and left_contour_center is not None and right_contour_center is not None:
        # In dual line mode, calculate steering to stay in the center between the lines
        left_x = left_contour_center[1]
        right_x = right_contour_center[1]
        
        # Calculate the midpoint between the two lines
        center_x = (left_x + right_x) / 2
        
        # Calculate angle to steer to the midpoint
        new_angle = center_x - rc.camera.get_width() / 2
        new_angle /= rc.camera.get_width() / 2
        
        # Use lane width to adjust speed (narrower = slower)
        lane_width = abs(right_x - left_x)
        lane_width_factor = lane_width / (rc.camera.get_width() / 2)
        
        # Print debug info about lane width
        if int(current_time * 5) % 5 == 0:  # Print every 1 second
            print(f"Lane width: {lane_width}px, factor: {lane_width_factor:.2f}")
    
    elif contour_center is not None:
        # Single line mode - modified angle calculation with offset
        center_x = contour_center[1]
        target_x = center_x + (line_offset * rc.camera.get_width() / 2)
        new_angle = target_x - rc.camera.get_width() / 2
        new_angle /= rc.camera.get_width() / 2

        # Add predictive steering if we have enough history
        if len(previous_centers) >= 3:
            # Calculate movement vector to predict where contour is heading
            dx = previous_centers[-1][1] - previous_centers[0][1]
            
            # Add prediction factor to steering
            prediction_factor = 0.4 * abs(speed)
            predicted_x = contour_center[1] + (dx * prediction_factor)
            predicted_angle = predicted_x - rc.camera.get_width() / 2
            predicted_angle /= rc.camera.get_width() / 2
            
            # Blend current position with prediction
            prediction_weight = min(0.4, 0.2 + (abs(speed) * 0.2))
            new_angle = ((1 - prediction_weight) * new_angle) + (prediction_weight * predicted_angle)
    
    elif recovery_mode:
        # Recovery mode - use last known good angle with some back-and-forth searching
        recovery_counter += delta_time
        
        # Start with the last known good angle
        new_angle = last_good_angle
        
        # Add a sinusoidal search pattern to try to find the line again
        search_amplitude = 0.3
        search_frequency = 1.5
        search_offset = math.sin(recovery_counter * search_frequency * 2 * math.pi) * search_amplitude
        new_angle += search_offset
        
        if int(recovery_counter * 10) % 5 == 0:
            print(f"RECOVERY MODE: Searching with angle {new_angle:.2f}")
    
    # Apply speed-dependent steering sensitivity
    if abs(speed) > 0.8:
        new_angle *= 0.75
    elif abs(speed) > 0.4:
        new_angle *= 0.85
    
    # Calculate target raw speed
    raw_speed = target_speed
    
    # Dual line mode specific speed adjustments
    if dual_line_mode and left_contour_center is not None and right_contour_center is not None:
        # Calculate lane width as a percentage of screen width
        lane_width = abs(right_contour_center[1] - left_contour_center[1])
        lane_width_percent = lane_width / rc.camera.get_width()
        
        # Adjust speed based on lane width - slower for very narrow or very wide lanes
        if lane_width_percent < 0.2:  # Very narrow lane
            raw_speed *= 0.6  # Significant speed reduction
        elif lane_width_percent > 0.7:  # Very wide lane (might be a detection error)
            raw_speed *= 0.7  # Moderate speed reduction
    
    # Secondary color speed adjustment (only if not in exclusive mode and not in dual line mode)
    if not dual_line_mode and not only_show_priority and contour_center is not None and current_color_index >= 0:
        # If we're tracking a color but the primary color isn't set to it,
        # it must be the secondary color, so go slightly slower
        detected_color = current_color_index
        if detected_color != current_color_index:
            raw_speed *= 0.9  # Go slightly slower for secondary color
    
    # Dynamic speed control based on contour tracking and turn severity
    if (dual_line_mode and (left_contour_center is None or right_contour_center is None)) or \
       (not dual_line_mode and contour_center is None):
        # Reduce speed when tracking is lost
        raw_speed = target_speed * 0.7
    elif not dual_line_mode and contour_area < 100:
        # Reduce speed when contour is small (less confident) - for single line mode only
        raw_speed = target_speed * 0.8
    
    # Apply speed caps based on turn sharpness
    turn_severity = abs(new_angle)
    if turn_severity > 0.8:
        speed_factor = rc_utils.remap_range(turn_severity, 0.8, 1.0, 0.7, 0.5)
        raw_speed = min(raw_speed, target_speed * speed_factor)
    elif turn_severity > 0.5:
        speed_factor = rc_utils.remap_range(turn_severity, 0.5, 0.8, 0.9, 0.7)
        raw_speed = min(raw_speed, target_speed * speed_factor)
    
    # Never go below minimum speed factor
    raw_speed = max(raw_speed, target_speed * min_speed_factor)
    
    # Speed change rate for responsive control
    speed_change_rate = 5.0 * delta_time
    if raw_speed > speed:
        speed = min(speed + speed_change_rate, raw_speed)
    else:
        speed = max(speed - speed_change_rate, raw_speed)
    
    # Smoothing for steering
    smoothing_factor = min(0.7, 0.3 + (abs(speed) * 0.4))
    angle = (smoothing_factor * prev_angle) + ((1 - smoothing_factor) * new_angle)
    prev_angle = angle

    # Ensure angle is within bounds
    if current_line_side == "right" or angle > 0:
        line_offset = 2
    elif current_line_side == "left" or angle < 0:
        line_offset = -2
    else:
        line_offset = 0
    angle = rc_utils.clamp(angle, -1.0, 1.0)

    # Set the new speed and angle of the racecar
    rc.drive.set_speed_angle(speed, angle)

def update_slow():
    """Updates the slow information."""

    # Check if a camera image is available
    if rc.camera.get_color_image() is None:
        print("X" * 10 + " (No image) " + "X" * 10)
    else:
        # Update display based on mode
        if dual_line_mode:
            if left_contour_center is None or right_contour_center is None:
                print("DUAL LINE MODE - " + "-" * 20 + " : Missing one or both lines")
            else:
                lane_width = abs(right_contour_center[1] - left_contour_center[1])
                mid_x = (left_contour_center[1] + right_contour_center[1]) // 2
                position_marker = ["-"] * 32
                position_marker[int(mid_x / 20)] = "|"
                print("DUAL LINE MODE - " + "".join(position_marker) + 
                      f" : lane_width = {lane_width}px, mid_x = {mid_x}")
        else:
            # Original single line logic
            if contour_center is None:
                print("-" * 32 + " : area = " + str(contour_area))
            else:
                color_name = "Unknown"
                if 0 <= current_color_index < len(COLORS):
                    color_name = ["Purple", "Orange"][current_color_index]
                s = ["-"] * 32
                s[int(contour_center[1] / 20)] = "|"
                print("".join(s) + f" : area = {contour_area} ({color_name}), " +
                     f"Following: {current_line_side.upper()} line")

# Main entry point
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()






