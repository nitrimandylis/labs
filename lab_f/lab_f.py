import sys
import cv2 as cv
import numpy as np

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

# Define a list of color ranges to search for in the camera image
COLORS = [
    # Red color range
    ((0, 80, 80),     (10, 255, 255)),   
    # Blue color range
    ((107, 81, 80),   (85, 255, 255)),   
    # Green color range
    ((100, 234, 150), (104, 255, 170)),  
    # Yellow color range
    ((90, 80, 80),    (120, 255, 255))   
]

# Initialize variables to store the speed, angle, contour center, and contour area
speed = 0.0  
angle = 0.0  
contour_center = None  
contour_area = 0  


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

    # Capture the current camera image
    image = rc.camera.get_color_image()

    # If the image is None, reset the contour information
    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Crop the image to focus on the floor
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

        # Initialize the contour variable
        contour = None

        # Iterate over the color ranges and search for contours
        for testingColor in COLORS:
            # Find contours within the current color range
            contours = rc_utils.find_contours(image, testingColor[0], testingColor[1])

            # If contours are found, get the largest contour and break out of the loop
            if contours:
                contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
                break    

        # If a contour is found, calculate its center and area
        if contour is not None:
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw the contour and its center on the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)

        # If no contour is found, reset the contour information
        else:
            contour_center = None
            contour_area = 0

        # Display the updated image
        rc.display.show_color_image(image)


# Define a function to initialize the racecar's speed and angle
def start():
    """
    Initializes the racecar's speed and angle.

    This function sets the initial speed and angle of the racecar to 0,
    and then sets the update slow time to 0.5 seconds.
    """
    global speed
    global angle

    speed = 0
    angle = 0

    # Set the initial speed and angle of the racecar
    rc.drive.set_speed_angle(speed, angle)

    # Set the update slow time to 0.5 seconds
    rc.set_update_slow_time(0.5)


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

    # Update the contour information
    update_contour()

    # If a contour is found, calculate the new angle based on the contour center
    if contour_center is not None:
        angle = contour_center[1] - rc.camera.get_width() / 2
        angle /= rc.camera.get_width() / 2

    # Get the forward and backward speed from the controller
    forwardSpeed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    backSpeed = rc.controller.get_trigger(rc.controller.Trigger.LEFT)

    # Calculate the new speed based on the forward and backward speed
    speed = forwardSpeed - backSpeed

    # Set the new speed and angle of the racecar
    rc.drive.set_speed_angle(speed, angle)

    # Check for controller input to print the speed and angle
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Check for controller input to print the contour information
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)


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
            s = ["-"] * 32
            s[int(contour_center[1] / 20)] = "|"
            print("".join(s) + " : area = " + str(contour_area))


# Check if this script is being run as the main program
if __name__ == "__main__":
    # Set the start, update, and update slow functions for the racecar
    rc.set_start_update(start, update, update_slow)
    # Start the racecar
    rc.go()