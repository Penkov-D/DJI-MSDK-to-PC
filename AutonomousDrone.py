from OpenDJI import OpenDJI

import keyboard
import cv2
import numpy as np
import time
import pyttsx3
import os # Import os module
from ai_processing import process_image_for_analysis

"""
In this example you can fly and see video from the drone in live!
Like a computer game, move the drone with the keyboard and see its image
on your computer screen!

    press F - to takeoff the drone.
    press R - to land the drone.
    press E - to enable control from keyboard (joystick disabled)
    press Q - to disable control from keyboard (joystick enabled)
    press X - to close the problam

    press W/S - to move up/down (ascent)
    press A/D - to rotate left/right (yaw control)
    press ↑/↓ - to move forward/backward (pitch)
    press ←/→ - to move left/right (roll)
"""

# IP address of the connected android device
IP_ADDR = "192.168.1.115"
# Set environment variable to bypass YOLO for this demo
os.environ["PROCESS_WITH_YOLO_FIRST"] = "true"

# The image from the drone can be quit big,
#  use this to scale down the image:
SCALE_FACTOR = 0.5

# Movement factors
MOVE_VALUE = 0.015
ROTATE_VALUE = 0.15

# Create blank frame
BLANK_FRAME = np.zeros((1080, 1920, 3))
BLNAK_FRAME = cv2.putText(BLANK_FRAME, "No Image", (200, 300),
                          cv2.FONT_HERSHEY_DUPLEX, 10,
                          (255, 255, 255), 15)

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Variable to track the last analysis time
last_analysis_time = 0

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:

    # Press 'x' to close the program
    print("Press 'x' to close the program")
    while not keyboard.is_pressed('x'):

        # Show image from the drone
        # Get frame
        frame = drone.getFrame()

        # What to do when no frame available
        if frame is None:
            frame_display = BLANK_FRAME  # Use a different variable for display to keep original 'frame'
        else:
            frame_display = frame
    
        # Resize frame - optional
        frame_display_resized = cv2.resize(frame_display, dsize = None,
                                           fx = SCALE_FACTOR,
                                           fy = SCALE_FACTOR)
        
        # Show frame
        cv2.imshow("Live video", frame_display_resized)
        cv2.waitKey(20)

        # AI Analysis every 5 seconds
        current_time = time.time()
        if current_time - last_analysis_time >= 5:
            print("Performing AI analysis...")
            # Use the original frame, not the blank or resized one for analysis
            frame_for_analysis = drone.getFrame() # Get a fresh frame for analysis
            
            if frame_for_analysis is not None:
                try:
                    analysis_result = process_image_for_analysis(frame_for_analysis)
                    print(f"AI Analysis Result: {analysis_result}")

                    # TTS output
                    if analysis_result and "No person detected" not in analysis_result:
                        tts_engine.say(analysis_result)
                        tts_engine.runAndWait()
                        # If a person is detected, take off and then land
                        print("Person detected, initiating takeoff and landing sequence.")
                        print(drone.takeoff(True))
                        # Adding a small delay for clarity, though takeoff/land might be blocking
                        print("Waiting for 10 seconds before landing...")
                        time.sleep(10) # Optional: adjust or remove as needed
                        print("Landing...")
                        print(drone.land(True))
                except Exception as e:
                    print(f"Error during AI analysis or TTS: {e}")
            else:
                print("No frame available for AI analysis at this moment.")
            last_analysis_time = current_time
        
        # Move the drone with the keyboard
        # Core variables
        yaw = 0.0       # Rotate, left horizontal stick
        ascent = 0.0    # Ascent, left vertical stick
        roll = 0.0      # Side movement, right horizontal stick
        pitch = 0.0     # Forward movement, right vertical stick

        # Set core variables based on key presses
        if keyboard.is_pressed('a'): yaw = -ROTATE_VALUE
        if keyboard.is_pressed('d'): yaw =  ROTATE_VALUE
        if keyboard.is_pressed('s'): ascent  = -MOVE_VALUE
        if keyboard.is_pressed('w'): ascent  =  MOVE_VALUE

        if keyboard.is_pressed('left'):  roll = -MOVE_VALUE
        if keyboard.is_pressed('right'): roll =  MOVE_VALUE
        if keyboard.is_pressed('down'):  pitch = -MOVE_VALUE
        if keyboard.is_pressed('up'):    pitch =  MOVE_VALUE

        # Send the movement command
        drone.move(yaw, ascent, roll, pitch)

        # Special commands
        if keyboard.is_pressed('f'): print(drone.takeoff(True))
        if keyboard.is_pressed('r'): print(drone.land(True))
        if keyboard.is_pressed('e'): print(drone.enableControl(True))
        if keyboard.is_pressed('q'): print(drone.disableControl(True))
