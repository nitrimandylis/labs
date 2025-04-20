

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import math

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

MIN_CONTOUR_AREA = 30

PURPLE = ((125, 100, 100), (140, 255, 255))
ORANGE = ((10, 150, 120), (25, 255, 255))

contour_center = None
contour_area = 0
speed = 0
angle = 0
prioritylist = [PURPLE, ORANGE]

recovery_mode = False
recovery_counter = 0
last_good_angle = 0
previous_centers = []
current_time = 0

largestcontour_center = None
secondcontour_center = None
generalcontour_center = None

accumulatedError = 0
lastError = 0

########################################################################################
# Functions
########################################################################################

def pid(Kp, Ki, Kd, target, current, dT):
    global accumulatedError
    global lastError
    error = target - current
    accumulatedError += error * dT
    accumulatedError = rc_utils.clamp(accumulatedError, -10, 10)
    deltaError = (error - lastError) / dT if dT > 0 else 0
    pTerm = Kp * error
    iTerm = Ki * accumulatedError
    dTerm = Kd * deltaError
    lastError = error
    return pTerm + iTerm + dTerm

def update_contour():
    global contour_center, contour_area
    global largestcontour_center, secondcontour_center, generalcontour_center
    global current_time

    image = rc.camera.get_color_image()
    
    if image is not None:
        crop_floor = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
        image = rc_utils.crop(image, crop_floor[0], crop_floor[1])
        
        largestcontour = None
        secondcontour = None
        largestcontour_center = (0, 0)
        secondcontour_center = (0, 0)
        generalcontour_center = (0, 0)
        
        for col in prioritylist:
            contours = rc_utils.find_contours(image, col[0], col[1])
            
            if not contours or len(contours) == 0:
                continue
                
            valid_contours = []
            for contour in contours:
                if contour is not None and cv.contourArea(contour) > MIN_CONTOUR_AREA:
                    valid_contours.append(contour)
            
            if len(valid_contours) == 0:
                continue
            
            sorted_contours = sorted(valid_contours, key=cv.contourArea, reverse=True)
            
            largestcontour = sorted_contours[0] if len(sorted_contours) > 0 else None
            secondcontour = sorted_contours[1] if len(sorted_contours) > 1 else None
            
            if largestcontour is not None:
                break
        
        if largestcontour is not None:
            largestcontour_center = rc_utils.get_contour_center(largestcontour)
            rc_utils.draw_contour(image, largestcontour, (0, 255, 0))
            if largestcontour_center is not None:
                rc_utils.draw_circle(image, largestcontour_center, (0, 255, 0))
                generalcontour_center = largestcontour_center
                contour_center = largestcontour_center
                contour_area = cv.contourArea(largestcontour)
        
        if secondcontour is not None:
            secondcontour_center = rc_utils.get_contour_center(secondcontour)
            rc_utils.draw_contour(image, secondcontour, (255, 0, 0))
            if secondcontour_center is not None:
                rc_utils.draw_circle(image, secondcontour_center, (255, 0, 0))
    
        rc.display.show_color_image(image)
    
    current_time += rc.get_delta_time()
    return largestcontour_center, secondcontour_center, generalcontour_center

def follow_two_lines():
    global speed, angle, last_good_angle, recovery_mode, recovery_counter, previous_centers
    
    cameraWidth = rc.camera.get_width()
    distancethreshold = 70
    delta_time = rc.get_delta_time()
    
    largestcontour_center, secondcontour_center, generalcontour_center = update_contour()
    
    has_largest = isinstance(largestcontour_center, tuple) and len(largestcontour_center) == 2 and largestcontour_center[0] != 0
    has_second = isinstance(secondcontour_center, tuple) and len(secondcontour_center) == 2 and secondcontour_center[0] != 0
    has_general = isinstance(generalcontour_center, tuple) and len(generalcontour_center) == 2 and generalcontour_center[0] != 0
    
    if has_largest and has_second:
        smallestx = min(largestcontour_center[1], secondcontour_center[1])
        largestx = max(largestcontour_center[1], secondcontour_center[1])
        center_point = (largestx + smallestx) / 2
        
        lane_width = largestx - smallestx
        print(f"Lane width: {lane_width} pixels")
        
        if (largestx - smallestx) > distancethreshold:
            target_point = center_point
            normalized_target = rc_utils.remap_range(target_point, 0, cameraWidth, -1, 1)
            angle = pid(0.5, 0.1, 0.2, 0, normalized_target, delta_time)
            speed = 0.3
        else:
            if center_point < (cameraWidth/2) - 30:
                angle = rc_utils.remap_range(center_point, 0, cameraWidth/2, 0.5, 0.1)
            elif center_point > (cameraWidth/2) + 30:
                angle = rc_utils.remap_range(center_point, cameraWidth/2, cameraWidth, -0.1, -0.5)
            speed = 0.2
        
        previous_centers.append((largestcontour_center[0], center_point))
        if len(previous_centers) > 4:
            previous_centers.pop(0)
            
        basic_angle = center_point - cameraWidth / 2
        basic_angle /= cameraWidth / 2
        last_good_angle = basic_angle
        
        recovery_mode = False
        recovery_counter = 0
            
    elif has_general:
        center_point = generalcontour_center[1]
        
        print(f"Following single line at x={center_point}")
        
        if center_point < cameraWidth/2:
            angle = rc_utils.remap_range(center_point, 0, cameraWidth/2, 0.5, 0.1)
        else:
            angle = rc_utils.remap_range(center_point, cameraWidth/2, cameraWidth, -0.1, -0.5)
        
        speed = 0.25
        
        previous_centers.append((generalcontour_center[0], center_point))
        if len(previous_centers) > 4:
            previous_centers.pop(0)
            
        basic_angle = center_point - cameraWidth / 2
        basic_angle /= cameraWidth / 2
        last_good_angle = basic_angle
        
        recovery_mode = False
        recovery_counter = 0
        
    else:
        enter_recovery_mode(delta_time)

def enter_recovery_mode(delta_time):
    global recovery_mode, recovery_counter, angle, speed, previous_centers
    
    if not recovery_mode:
        recovery_mode = True
        recovery_counter = 0
        print("Entering RECOVERY MODE - no lines detected")
    
    if len(previous_centers) > 0:
        if len(previous_centers) > 1:
            previous_centers.pop(0)
    
    recovery_counter += delta_time
    
    angle = last_good_angle
    
    search_amplitude = 0.3
    search_frequency = 1.5
    search_offset = math.sin(recovery_counter * search_frequency * 2 * math.pi) * search_amplitude
    angle += search_offset
    
    if int(recovery_counter * 10) % 5 == 0:
        print(f"RECOVERY MODE: Searching with angle {angle:.2f}")
    
    speed = 0.15
    
    if recovery_counter > 2.0:
        recovery_counter = 0

def start():
    global recovery_mode, recovery_counter, last_good_angle, previous_centers
    global accumulatedError, lastError, speed, angle, current_time
    
    accumulatedError = 0
    lastError = 0
    current_time = 0
    recovery_mode = False
    recovery_counter = 0
    last_good_angle = 0
    previous_centers = []
    speed = 0
    angle = 0
    
    rc.drive.set_speed_angle(0, 0)
    rc.drive.set_max_speed(0.5)
    
    print(
        ">> Lab F - Two Line Following Challenge\n"
        "\n"
        "Controls:\n"
        "   X button = set ORANGE as primary color\n"
        "   Y button = set PURPLE as primary color"
    )
    
    rc.set_update_slow_time(0.5)

def update():
    global prioritylist, angle, speed
    
    if rc.controller.was_pressed(rc.controller.Button.X):
        prioritylist = [ORANGE, PURPLE]
        print("Set ORANGE as primary color")
    
    if rc.controller.was_pressed(rc.controller.Button.Y):
        prioritylist = [PURPLE, ORANGE]
        print("Set PURPLE as primary color")
    
    follow_two_lines()
    
    if angle > 0:
        angle += 0.3
    elif angle < 0:
        angle -= 0.3
    angle = rc_utils.clamp(angle, -1, 1)
    
    speed_factor = 1.0 - abs(angle) * 1.5
    calculate_speed = speed * max(0.3, speed_factor)
    
    calculate_speed = rc_utils.clamp(calculate_speed, 0.1, 1)
    
    rc.drive.set_speed_angle(0.8, angle)
    
    print(f"Speed: {calculate_speed:.2f}, Angle: {angle:.2f}")

def update_slow():
    if rc.camera.get_color_image() is None:
        print("WARNING: No camera image available!")
    
    color_names = ["PURPLE", "ORANGE"]
    if prioritylist[0] == PURPLE:
        print(f"Currently prioritizing: {color_names[0]}, then {color_names[1]}")
    else:
        print(f"Currently prioritizing: {color_names[1]}, then {color_names[0]}")
    
    if recovery_mode:
        print(f"Mode: RECOVERY MODE (searching for {recovery_counter:.1f}s)")
    else:
        print("Mode: Two Line Following")
        
    print(f"Camera crop: Starting at 360px from top")

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()