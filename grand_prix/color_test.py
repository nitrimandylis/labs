"""
Color Testing Script for Racecar

This script helps test and visualize HSV color ranges for the racecar.
Use it to fine-tune color detection ranges.

Instructions:
- Press 'q' to quit
- Press 's' to save a snapshot
- Use trackbars to adjust HSV ranges

"""

import sys
import os
import cv2 as cv
import numpy as np

# Add racecar library to path
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

# Initialize racecar
rc = racecar_core.create_racecar()

# Current HSV color ranges
PURPLE = ((125, 100, 100), (145, 255, 255))
ORANGE = ((5, 100, 100), (25, 255, 255))
GREEN = ((60, 200, 200), (80, 255, 255))  # Neon green

# Window names
MAIN_WINDOW = "Color Test"
CONTROLS_WINDOW = "HSV Controls"
MASKS_WINDOW = "Color Masks"

# Create trackbar variables with initial values from PURPLE
h_min = PURPLE[0][0]
s_min = PURPLE[0][1]
v_min = PURPLE[0][2]
h_max = PURPLE[1][0]
s_max = PURPLE[1][1]
v_max = PURPLE[1][2]

def create_controls():
    """Create trackbar controls for adjusting HSV values"""
    cv.namedWindow(CONTROLS_WINDOW)
    cv.createTrackbar("H min", CONTROLS_WINDOW, h_min, 179, lambda x: None)
    cv.createTrackbar("S min", CONTROLS_WINDOW, s_min, 255, lambda x: None)
    cv.createTrackbar("V min", CONTROLS_WINDOW, v_min, 255, lambda x: None)
    cv.createTrackbar("H max", CONTROLS_WINDOW, h_max, 179, lambda x: None)
    cv.createTrackbar("S max", CONTROLS_WINDOW, s_max, 255, lambda x: None)
    cv.createTrackbar("V max", CONTROLS_WINDOW, v_max, 255, lambda x: None)

def read_trackbar_values():
    """Read current values from trackbars"""
    global h_min, s_min, v_min, h_max, s_max, v_max
    
    h_min = cv.getTrackbarPos("H min", CONTROLS_WINDOW)
    s_min = cv.getTrackbarPos("S min", CONTROLS_WINDOW)
    v_min = cv.getTrackbarPos("V min", CONTROLS_WINDOW)
    h_max = cv.getTrackbarPos("H max", CONTROLS_WINDOW)
    s_max = cv.getTrackbarPos("S max", CONTROLS_WINDOW)
    v_max = cv.getTrackbarPos("V max", CONTROLS_WINDOW)
    
    return ((h_min, s_min, v_min), (h_max, s_max, v_max))

def save_snapshot(image, hsv_image, mask):
    """Save the current images to disk"""
    timestamp = rc.get_delta_time()
    cv.imwrite(f"color_test_rgb_{timestamp:.0f}.jpg", image)
    cv.imwrite(f"color_test_hsv_{timestamp:.0f}.jpg", hsv_image)
    cv.imwrite(f"color_test_mask_{timestamp:.0f}.jpg", mask)
    print(f"Snapshot saved with timestamp {timestamp:.0f}")

def display_color_info(hsv_range):
    """Display HSV range information on console"""
    print("\n==== HSV COLOR RANGE INFORMATION ====")
    print(f"Current HSV Range: {hsv_range}")
    print(f"Python format: (({hsv_range[0][0]}, {hsv_range[0][1]}, {hsv_range[0][2]}), ({hsv_range[1][0]}, {hsv_range[1][1]}, {hsv_range[1][2]}))")
    
    # Create a dummy BGR color from the middle of the HSV range to show approximate RGB value
    h_mid = (hsv_range[0][0] + hsv_range[1][0]) // 2
    s_mid = (hsv_range[0][1] + hsv_range[1][1]) // 2
    v_mid = (hsv_range[0][2] + hsv_range[1][2]) // 2
    
    # Create a 1x1 HSV image and convert to BGR
    hsv_sample = np.uint8([[[h_mid, s_mid, v_mid]]])
    bgr_sample = cv.cvtColor(hsv_sample, cv.COLOR_HSV2BGR)
    b, g, r = bgr_sample[0][0]
    
    print(f"Approximate RGB equivalent: R:{r}, G:{g}, B:{b}")
    print("==== END COLOR INFORMATION ====\n")

def test_predefined_colors(image):
    """Test predefined color ranges"""
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Create masks for predefined colors
    purple_mask = cv.inRange(hsv_image, PURPLE[0], PURPLE[1])
    orange_mask = cv.inRange(hsv_image, ORANGE[0], ORANGE[1])
    
    # Apply masks to original image
    purple_result = cv.bitwise_and(image, image, mask=purple_mask)
    orange_result = cv.bitwise_and(image, image, mask=orange_mask)
    
    # Create color detection display
    h, w, _ = image.shape
    predefined_display = np.zeros((h, w*3, 3), dtype=np.uint8)
    
    # Place images side by side
    predefined_display[:, :w] = image
    predefined_display[:, w:w*2] = purple_result
    predefined_display[:, w*2:] = orange_result
    
    # Add labels
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(predefined_display, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv.putText(predefined_display, "Purple", (w+10, 30), font, 1, (255, 0, 255), 2)
    cv.putText(predefined_display, "Orange", (w*2+10, 30), font, 1, (0, 165, 255), 2)
    
    # Add HSV ranges for predefined colors at the bottom
    purple_text = f"PURPLE: H:{PURPLE[0][0]}-{PURPLE[1][0]}, S:{PURPLE[0][1]}-{PURPLE[1][1]}, V:{PURPLE[0][2]}-{PURPLE[1][2]}"
    orange_text = f"ORANGE: H:{ORANGE[0][0]}-{ORANGE[1][0]}, S:{ORANGE[0][1]}-{ORANGE[1][1]}, V:{ORANGE[0][2]}-{ORANGE[1][2]}"
    cv.putText(predefined_display, purple_text, (w+10, h-40), font, 0.5, (255, 0, 255), 1)
    cv.putText(predefined_display, orange_text, (w*2+10, h-40), font, 0.5, (0, 165, 255), 1)
    
    return predefined_display

def start():
    """Initialize the testing"""
    create_controls()
    print("Color testing initialized. Press 'q' to quit, 's' to save snapshot.")
    print(f"Initial PURPLE range: {PURPLE}")
    print(f"Initial ORANGE range: {ORANGE}")

def update():
    """Main update function"""
    # Get image from camera
    image = rc.camera.get_color_image()
    
    if image is None:
        print("No image from camera")
        return
        
    # Convert to OpenCV BGR format if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
    
    # Get current HSV range from trackbars
    hsv_range = read_trackbar_values()
    
    # Convert to HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Create mask with current range
    mask = cv.inRange(hsv_image, hsv_range[0], hsv_range[1])
    
    # Apply mask to original image
    result = cv.bitwise_and(image, image, mask=mask)
    
    # Create display image with original, mask, and result
    h, w, _ = image.shape
    display = np.zeros((h, w*3, 3), dtype=np.uint8)
    
    # Place images side by side
    display[:, :w] = image
    # Convert mask to BGR for display
    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    display[:, w:w*2] = mask_bgr
    display[:, w*2:] = result
    
    # Add labels
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(display, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv.putText(display, "Mask", (w+10, 30), font, 1, (255, 255, 255), 2)
    cv.putText(display, "Result", (w*2+10, 30), font, 1, (255, 255, 255), 2)
    
    # Add current HSV values to display
    hsv_text = f"HSV Range: H:{hsv_range[0][0]}-{hsv_range[1][0]}, S:{hsv_range[0][1]}-{hsv_range[1][1]}, V:{hsv_range[0][2]}-{hsv_range[1][2]}"
    cv.putText(display, hsv_text, (10, h-20), font, 0.6, (0, 255, 255), 2)
    
    # Show the image
    cv.imshow(MAIN_WINDOW, display)
    
    # Show predefined colors test
    predefined_display = test_predefined_colors(image)
    cv.imshow(MASKS_WINDOW, predefined_display)
    
    # Process key presses
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        rc.set_update_slow_time(0)
        rc.drive.stop()
        sys.exit()
    elif key == ord('s'):
        save_snapshot(image, hsv_image, mask)
        display_color_info(hsv_range)
    
    # Update live data display
    rc.display.show_color_image(result)

def update_slow():
    """Slower update function for logging info"""
    hsv_range = read_trackbar_values()
    
    # Create a header with clear separation
    print("\n---------------------------------------")
    print("CURRENT HSV RANGE (updated every second):")
    print(f"H: {hsv_range[0][0]} to {hsv_range[1][0]}")
    print(f"S: {hsv_range[0][1]} to {hsv_range[1][1]}")
    print(f"V: {hsv_range[0][2]} to {hsv_range[1][2]}")
    print("Python format:")
    print(f"(({hsv_range[0][0]}, {hsv_range[0][1]}, {hsv_range[0][2]}), ({hsv_range[1][0]}, {hsv_range[1][1]}, {hsv_range[1][2]}))")
    print("---------------------------------------")
    
    # Create a small color swatch image to display the current color
    swatch_size = 100
    swatch = np.zeros((swatch_size, swatch_size, 3), dtype=np.uint8)
    
    # Get middle values of current range
    h_mid = (hsv_range[0][0] + hsv_range[1][0]) // 2
    s_mid = (hsv_range[0][1] + hsv_range[1][1]) // 2
    v_mid = (hsv_range[0][2] + hsv_range[1][2]) // 2
    
    # Fill the swatch with the HSV color
    hsv_swatch = np.zeros((swatch_size, swatch_size, 3), dtype=np.uint8)
    hsv_swatch[:, :] = [h_mid, s_mid, v_mid]
    
    # Convert to BGR for display
    bgr_swatch = cv.cvtColor(hsv_swatch, cv.COLOR_HSV2BGR)
    
    # Add text with current values
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(bgr_swatch, f"H:{h_mid}", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv.putText(bgr_swatch, f"S:{s_mid}", (10, 40), font, 0.5, (255, 255, 255), 1)
    cv.putText(bgr_swatch, f"V:{v_mid}", (10, 60), font, 0.5, (255, 255, 255), 1)
    
    # Display the swatch
    cv.imshow("Current Color", bgr_swatch)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.set_update_slow_time(1.0)
    rc.go() 