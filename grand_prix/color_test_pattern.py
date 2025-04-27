"""
Color Test Pattern Generator for Racecar

This script creates test pattern images with specific RGB colors,
especially focusing on RGB(128, 0, 255) purple for color detection testing.

Usage:
- Run the script to generate a color test pattern
- The pattern includes RGB(128, 0, 255) purple and other colors
- The image is saved to disk
"""

import cv2 as cv
import numpy as np
import time

# Define constants
WIDTH = 640
HEIGHT = 480
CELL_SIZE = 80
COLORS = [
    # Format: (B, G, R) (OpenCV uses BGR color order)
    (255, 0, 128),  # Purple RGB(128, 0, 255) - Our target purple color
    (0, 127, 255),  # Orange RGB(255, 127, 0) - Our target orange color
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 128, 128),# Gray
    (255, 255, 255),# White
    (0, 0, 0)       # Black
]

def create_test_pattern():
    """Create a color test pattern image"""
    # Create blank image
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Create a grid of color blocks
    for i, color in enumerate(COLORS):
        # Calculate row and column
        row = i // (WIDTH // CELL_SIZE)
        col = i % (WIDTH // CELL_SIZE)
        
        # Calculate top-left corner
        x1 = col * CELL_SIZE
        y1 = row * CELL_SIZE
        
        # Calculate bottom-right corner
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE
        
        # Draw the colored rectangle
        cv.rectangle(image, (x1, y1), (x2, y2), color, -1)
        
        # Add text label - RGB values
        r, g, b = color[2], color[1], color[0]  # Convert from BGR to RGB for display
        text = f"R:{r} G:{g} B:{b}"
        
        # Choose text color based on background brightness
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        # Position text
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + (CELL_SIZE - text_size[0]) // 2
        text_y = y1 + (CELL_SIZE + text_size[1]) // 2
        
        cv.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Add a special highlight around the target colors
    cv.rectangle(image, (0, 0), (CELL_SIZE, CELL_SIZE), (255, 255, 255), 2)
    cv.putText(image, "PURPLE", (5, CELL_SIZE - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Highlight the target orange (should be in the second position)
    row = 1 // (WIDTH // CELL_SIZE)
    col = 1 % (WIDTH // CELL_SIZE)
    x1 = col * CELL_SIZE
    y1 = row * CELL_SIZE
    x2 = x1 + CELL_SIZE
    y2 = y1 + CELL_SIZE
    cv.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv.putText(image, "ORANGE", (x1 + 5, y2 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create a gradient of purples
    # This creates a strip at the bottom with varying shades of purple
    gradient_height = 50
    gradient_y = HEIGHT - gradient_height * 2
    
    for x in range(WIDTH):
        # Vary hue slightly around the purple we want
        r_val = 128 + int(127 * (x / WIDTH) - 60)
        r_val = max(0, min(255, r_val))
        
        # Keep blue at maximum
        b_val = 255
        
        # Vary green slightly
        g_val = int(50 * (x / WIDTH))
        g_val = max(0, min(255, g_val))
        
        # Draw vertical line with this color
        color = (b_val, g_val, r_val)  # BGR format
        cv.line(image, (x, gradient_y), (x, gradient_y + gradient_height), color, 1)
    
    cv.putText(image, "Purple Gradient", (10, gradient_y + gradient_height // 2), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create a gradient of oranges
    gradient_y = HEIGHT - gradient_height
    
    for x in range(WIDTH):
        # Vary orange hue slightly around the target orange
        r_val = 255  # Keep red at maximum
        
        # Vary green around 127
        g_val = int(127 + (x / WIDTH - 0.5) * 50)
        g_val = max(0, min(255, g_val))
        
        # Vary blue slightly (keep minimal for orange)
        b_val = int(50 * (x / WIDTH))
        b_val = max(0, min(100, b_val))
        
        # Draw vertical line with this color
        color = (b_val, g_val, r_val)  # BGR format
        cv.line(image, (x, gradient_y), (x, HEIGHT), color, 1)
    
    cv.putText(image, "Orange Gradient", (10, HEIGHT - gradient_height // 2), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def main():
    """Main function"""
    # Create the test pattern
    test_pattern = create_test_pattern()
    
    # Save the image
    timestamp = int(time.time())
    filename = f"color_test_pattern_{timestamp}.jpg"
    cv.imwrite(filename, test_pattern)
    print(f"Test pattern saved as {filename}")
    
    # Display the image
    cv.imshow("Color Test Pattern", test_pattern)
    print("Press any key to exit...")
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main() 