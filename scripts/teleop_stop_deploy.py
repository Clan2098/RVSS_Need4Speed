#!/usr/bin/env python
import time
import sys
import os
import cv2
import numpy as np
import argparse
from datetime import datetime
from stop_detector import StopSignDetector
from controller import Controller
from teleop import Teleop

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot


parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--im_num', type = int, default = 0)
# parser.add_argument('--folder', type = str, default = 'train')
args = parser.parse_args()

# if not os.path.exists(script_path+"/../data/"+args.folder):
#     data_path = script_path.replace('scripts', 'data')
#     print(f'Folder "{args.folder}" in path {data_path} does not exist. Please create it.')
#     exit()

bot = PiBot(ip=args.ip)

# Stop the robot
bot.setVelocity(0, 0)

# Configuration
CONSECUTIVE_FRAMES_REQUIRED = 3
STOP_DURATION = 1.0  # seconds
COOLDOWN_DURATION = 5.0  # seconds

# Stop detection state
consecutive_stop_count = 0
stop_start_time = None
ignore_stop_until = None

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

im_number = args.im_num
controller = Controller()
stop_sign_detector = StopSignDetector()

# Start in paused state
controller.stopped = True

# Define what to do when an error occurs
def on_error(_error):
    bot.setVelocity(0, 0)

# Define what to do when the user toggles the pause
def on_toggle_stop():
    controller.toggle_stop()

teleop = Teleop(on_error, on_toggle_stop)
teleop.start()

try:
    while teleop.continue_running:
        # Get an image from the robot
        current_time = time.time()
        img = bot.getImage()

        # Check if we're in cooldown period
        in_cooldown = (ignore_stop_until is not None and current_time < ignore_stop_until)

        # Handle active stop (takes priority over everything)
        if stop_start_time is not None:
            if current_time - stop_start_time < STOP_DURATION:
                # Still in stop period
                bot.setVelocity(0, 0)
                print(f"Stopped... {STOP_DURATION - (current_time - stop_start_time):.1f}s remaining")
                time.sleep(0.1)
                continue
            else:
                # Stop period ended
                print("Resuming motion...")
                stop_start_time = None
        
        # Only detect stops if NOT paused and NOT in cooldown
        stop_detected = False
        if not in_cooldown:
            stop_detected = stop_sign_detector.detect(img)
            
            if stop_detected:
                consecutive_stop_count += 1
                print(f"Stop sign detected! Consecutive count: {consecutive_stop_count}")
            else:
                consecutive_stop_count = 0
            
            # Trigger stop if threshold reached
            if consecutive_stop_count >= CONSECUTIVE_FRAMES_REQUIRED:
                print(f"STOP SIGN CONFIRMED ({consecutive_stop_count} frames)! Stopping for {STOP_DURATION}s...")
                stop_start_time = current_time
                ignore_stop_until = current_time + STOP_DURATION + COOLDOWN_DURATION
                consecutive_stop_count = 0
                bot.setVelocity(0, 0)
                time.sleep(0.1)
                continue
        else:
            # Reset counter if paused or in cooldown
            consecutive_stop_count = 0

        left, right = controller(teleop.label)
        controller_angle = controller.angle

        bot.setVelocity(left, right)
        print(left, right)
        print(stop_detected)

        time.sleep(0.1)  # Small delay to reduce CPU usage

    # Clean up
    bot.setVelocity(0, 0)
    teleop.stop()
    print("Script ended")


except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
    teleop.stop()