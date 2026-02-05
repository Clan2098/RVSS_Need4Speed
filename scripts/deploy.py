#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

from train_net import Net # TODO: should we change the name to avoid confusions?
from train_stop import StopNet
from controller import Controller


########## CUSTOM FUNCTIONS ##########
def preprocess_image(img, transform):
    """Preprocess images"""
    img_cropped = img[120:, :, :]
    img_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
    img_tensor = transform(img_hsv)
    return img_tensor.unsqueeze(0) # Add batch dimension

# Define what to do when an error occurs
def on_error(_error):
    bot.setVelocity(0, 0)
#####################################


parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--im_num', type = int, default = 0)
parser.add_argument('--folder', type = str, default = 'train')
args = parser.parse_args()

bot = PiBot(ip=args.ip)

# stop the robot
bot.setVelocity(0, 0)

# INITIALISE NETWORK HERE
stop_net = StopNet()
steering_net = Net()

# LOAD NETWORK WEIGHTS HERE
stop_net.load_state_dict(torch.load('stop_net.pth'))
stop_net.eval()

steering_net.load_state_dict(torch.load('steer_net.pth'))
steering_net.eval()

# DEFINE ANY IMAGE TRANSFORMS HERE, they must be the same as those used during training
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((40, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

# Configuration
CONSECUTIVE_FRAMES_REQUIRED = 2  # Easy to adjust
STOP_DURATION = 5.0  # seconds
COOLDOWN_DURATION = 15.0  # seconds

# Stop detection state
consecutive_stop_count = 0
stop_start_time = None
ignore_stop_until = None

# Countdown before beginning
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

try:
    angle = 0
    while True:
        # Get an image from the the robot
        current_time = time.time()
        img = bot.getImage()

        # Apply any necessary image transforms
        img_tensor = preprocess_image(img, transform)

        # Check if we're in cooldown period
        in_cooldown = (ignore_stop_until is not None and current_time < ignore_stop_until)

        # Detect stop sign
        stop_detected = False

        # If in cooldown, skip stop detection
        if not in_cooldown:
            with torch.no_grad():
                stop_output = stop_net(img_tensor)
                stop_detected = (torch.max(stop_output, 1)[1].item() == 1) # torch.max looks for the maximum value along the specified dimension (in this case, dimension 1 which corresponds to the class scores) and returns both the maximum value and its index. The [1] is used to get the index of the maximum value, which corresponds to the predicted class. We then check if this predicted class index is equal to 1, which indicates that a stop sign is detected.

            if stop_detected:
                consecutive_stop_count += 1
                print(f"Stop sign detected! Consecutive count: {consecutive_stop_count}")
            else:
                consecutive_stop_count = 0

            # Trigger stop if threshold reached
            if consecutive_stop_count >= CONSECUTIVE_FRAMES_REQUIRED and stop_start_time is None:
                print(f"STOP SIGN DETECTED ({consecutive_stop_count} frames)! Stopping for {STOP_DURATION}s...")
                stop_start_time = current_time
                ignore_stop_until = current_time + COOLDOWN_DURATION
                consecutive_stop_count = 0
        
            # Handle active stop
            if stop_start_time is not None:
                if current_time - stop_start_time < STOP_DURATION:
                    bot.setVelocity(0, 0)
                    continue
                else:
                    print("Resuming motion...")
                    stop_start_time = None
        
        # Pass image through network get a steering prediction
        with torch.no_grad():
            steering_output = steering_net(img_tensor)

        #TODO: convert prediction into a meaningful steering angle
        
        angle = 0

        Kd = 20 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 20 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
