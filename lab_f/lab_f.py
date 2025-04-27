import sys
import cv2 as cv
import numpy as np
import math

# Import the racecar_core and racecar_utils libraries, which provide functionality for interacting with the racecar
sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

# Create a racecar object, which is used to interact with the physical racecar
rc = racecar_core.create_racecar()

# Define the minimum contour area required for a contour to be considered valid
MIN_CONTOUR_AREA = 30

# Define the crop region for the camera image, which is used to focus on the floor in front of the racecar
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# Define a list of color ranges to search for in the camera image with much higher minimum saturation
# to better distinguish from shadows
COLORS = [
    # Red color range - much higher saturation minimum
    ((0, 150, 120),     (10, 255, 255)),   
    # Green color range - much wider and more forgiving
    ((40, 100, 100),    (85, 255, 255)),  
    # Blue color range - much higher saturation minimum
    ((90, 150, 120),    (120, 255, 255))   
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

# Define a function to preprocess the image to reduce shadow effects
def preprocess_image(image):
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
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clipLimit
    v = clahe.apply(v)
    
    # Enhance saturation slightly to make colors pop more compared to shadows
    s = cv.multiply(s, 1.3)  # Increase saturation by 30%
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # Merge channels back
    hsv = cv.merge([h, s, v])
    
    return hsv

# Define a function to create a shadow mask for the image
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

# Define a function to safely apply a shadow overlay to an image
def apply_shadow_overlay(image, shadow_mask, opacity=0.3):
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

# Define a function to update the contour information based on the current camera image
def update_contour():
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
                display_img = apply_shadow_overlay(display_img, shadow_mask, 0.3)
            
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
            preprocessed = preprocess_image(main_image)
            shadow_mask = create_shadow_mask(preprocessed)
            
            # Create recovery display
            display_img = main_image.copy()
            
            # Add shadow overlay using our safe function
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


# Define a function to initialize the racecar's speed and angle
def start():
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
    
    print("Racecar initialized with FULL SPEED and SHADOW DETECTION.")
    print("CONTROLS:")
    print("X/Y: Adjust speed")
    print("LEFT_BUMPER: Prioritize RED")
    print("RIGHT_BUMPER: Prioritize GREEN")
    print("A: Prioritize BLUE")
    print("B: Reset color priority (track all colors equally)")
    print("Shadows will be shown in red overlay in debug mode.")


# Define a function to update the racecar's speed and angle based on the current contour information
def update():
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

    # Calculate delta time for smoother motion and timing
    current_time = rc.get_delta_time()
    delta_time = current_time - last_update_time
    last_update_time = current_time

    # Update the contour information
    update_contour()

    # Speed control with controller
    if rc.controller.was_pressed(rc.controller.Button.X):
        target_speed -= 0.1
        target_speed = max(0.4, target_speed)  # Minimum 0.4 instead of 0.2
        print(f"Target speed decreased to {target_speed:.1f}")
    
    if rc.controller.was_pressed(rc.controller.Button.Y):
        target_speed += 0.1
        target_speed = min(1.0, target_speed)
        print(f"Target speed increased to {target_speed:.1f}")


    # Color priority selection with different buttons
    if rc.controller.was_pressed(rc.controller.Button.LB):
        current_color_index = 0  # Prioritize Red
        print("Prioritizing RED color")
    
    if rc.controller.was_pressed(rc.controller.Button.RB):
        current_color_index = 1  # Prioritize Green
        print("Prioritizing GREEN color")
    
    if rc.controller.was_pressed(rc.controller.Button.A):
        current_color_index = 2  # Prioritize Blue
        print("Prioritizing BLUE color")
        
    if rc.controller.was_pressed(rc.controller.Button.B):
        if current_color_index >= 0:
            # Toggle "only show priority" mode when a priority is set
            only_show_priority = not only_show_priority
            print(f"{'ONLY' if only_show_priority else 'PRIORITIZING'} {['Red', 'Green', 'Blue'][current_color_index]} color")
        else:
            # Reset color tracking when no priority is set
            current_color_index = -1
            only_show_priority = False
            print("Color tracking reset - will search for all colors")

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
        # Recovery mode - use last known good angle with some back-and-forth searching
        recovery_counter += delta_time
        
        # Start with the last known good angle
        new_angle = last_good_angle
        
        # Add a sinusoidal search pattern to try to find the line again
        search_amplitude = 0.3  # How far to search left/right
        search_frequency = 1.5  # How fast to search (increased from 1.0)
        search_offset = math.sin(recovery_counter * search_frequency * 2 * math.pi) * search_amplitude
        new_angle += search_offset
        
        # Don't print every frame to reduce console spam
        if int(recovery_counter * 10) % 5 == 0:
            print(f"RECOVERY MODE: Searching with angle {new_angle:.2f}")
    
    # Apply speed-dependent steering sensitivity (less reduction)
    if abs(speed) > 0.8:
        # Reduce steering sensitivity at high speeds to prevent oscillation
        new_angle *= 0.75  # Less reduction (was 0.65)
    elif abs(speed) > 0.4:
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
        speed_factor = rc_utils.remap_range(turn_severity, 0.8, 1.0, 0.7, 0.5)  # Higher min value
        raw_speed = min(raw_speed, target_speed * speed_factor)
    elif turn_severity > 0.5:  # Was 0.4, now 0.5
        # Less slowdown for medium turns
        speed_factor = rc_utils.remap_range(turn_severity, 0.5, 0.8, 0.9, 0.7)  # Higher min value
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
    if speed < 0.5:
        speed += 0.1
    speed = rc_utils.clamp(speed, -1.0, 1.0)

    speed = rc_utils.clamp(speed, 0.0, 1.0)
    angle = rc_utils.clamp(angle, -1.0, 1.0)
    # Set the new speed and angle of the racecar
    print("speed: ", speed)
    print("angle: ", angle)
    rc.drive.set_speed_angle(speed, angle)


    # Display status information when holding down certain buttons


# Define a function to update the slow information
def update_slow():
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


# Check if this script is being run as the main program
if __name__ == "__main__":
    # Set the start, update, and update slow functions for the racecar
    rc.set_start_update(start, update, update_slow)
    # Start the racecar
    rc.go()