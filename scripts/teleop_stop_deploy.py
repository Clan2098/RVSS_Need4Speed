#!/usr/bin/env python
import time
import sys
import os
import cv2
import numpy as np
import argparse
from datetime import datetime

from controller import Controller
from teleop import Teleop

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--im_num', type = int, default = 0)
parser.add_argument('--folder', type = str, default = 'train')
args = parser.parse_args()

if not os.path.exists(script_path+"/../data/"+args.folder):
    data_path = script_path.replace('scripts', 'data')
    print(f'Folder "{args.folder}" in path {data_path} does not exist. Please create it.')
    exit()

bot = PiBot(ip=args.ip)

# stop the robot
bot.setVelocity(0, 0)

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
        img = bot.getImage()

        # TODO: Pass image through stop net to predict stop sign

        left, right = controller(teleop.label)
        controller_angle = controller.angle

        # TODO: Logic for controller if stop detected
        
        bot.setVelocity(left, right)

        time.sleep(0.1)  # Small delay to reduce CPU usage

    # Clean up
    bot.setVelocity(0, 0)
    teleop.stop()
    print("Script ended")


except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
    teleop.stop()