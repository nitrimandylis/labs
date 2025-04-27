"""
RGB to HSV Converter for Racecar

This utility script helps convert RGB color values to HSV values used by OpenCV.
It's useful for determining the HSV range for a specific RGB color.

Usage:
python rgb_to_hsv.py [r] [g] [b]

Examples:
python rgb_to_hsv.py 128 0 255  # Convert RGB(128, 0, 255) to HSV - Purple
python rgb_to_hsv.py 255 127 0   # Convert RGB(255, 127, 0) to HSV - Orange
"""

import sys
import cv2 as cv
import numpy as np

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV using OpenCV"""
    # Create a 1-pixel BGR image (OpenCV uses BGR order)
    rgb = np.uint8([[[b, g, r]]])
    
    # Convert to HSV
    hsv = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
    h, s, v = hsv[0][0]
    
    return h, s, v

def suggest_range(h, s, v):
    """Suggest a suitable HSV range for the given color"""
    # For Hue, suggest a tight range of ±2
    h_min = max(0, h - 2)
    h_max = min(179, h + 2)
    
    # For Saturation, suggest a range based on the value
    s_min = max(0, s - 10)
    s_max = 255
    
    # For Value, suggest a range based on the value
    v_min = max(0, v - 10)
    v_max = 255
    
    return (h_min, s_min, v_min), (h_max, s_max, v_max)

def main():
    """Main function"""
    if len(sys.argv) != 4:
        print("Usage: python rgb_to_hsv.py [r] [g] [b]")
        print("Example: python rgb_to_hsv.py 255 127 0")
        
        # Default to the target orange color
        r, g, b = 255, 127, 0
        print("\nUsing default orange RGB(255, 127, 0)")
    else:
        # Parse command line arguments
        r = int(sys.argv[1])
        g = int(sys.argv[2])
        b = int(sys.argv[3])
    
    # Convert RGB to HSV
    h, s, v = rgb_to_hsv(r, g, b)
    
    # Suggest a range
    hsv_min, hsv_max = suggest_range(h, s, v)
    
    # Print results
    print(f"RGB({r}, {g}, {b}) converts to HSV({h}, {s}, {v})")
    print("\nSuggested HSV range for OpenCV:")
    print(f"Lower bound: ({hsv_min[0]}, {hsv_min[1]}, {hsv_min[2]})")
    print(f"Upper bound: ({hsv_max[0]}, {hsv_max[1]}, {hsv_max[2]})")
    print("\nPython tuple format:")
    print(f"(({hsv_min[0]}, {hsv_min[1]}, {hsv_min[2]}), ({hsv_max[0]}, {hsv_max[1]}, {hsv_max[2]}))")
    
    # Create a sample image to visualize the color
    sample_size = 200
    sample = np.zeros((sample_size, sample_size, 3), dtype=np.uint8)
    sample[:] = (b, g, r)  # OpenCV uses BGR order
    
    # Display the sample
    cv.imshow(f"RGB({r}, {g}, {b})", sample)
    print("\nPress any key to exit...")
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main() 