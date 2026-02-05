#!/usr/bin/env python3
import time
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse

from stop_detector import StopSignDetector
from controller import Controller

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot
from steer_labels import LABELS, steering_to_class, label_to_class, LABEL_TO_ANGLE
from controller import Controller


parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
args = parser.parse_args()

bot = PiBot(ip=args.ip)

controller = Controller()

# Configuration
CONSECUTIVE_FRAMES_REQUIRED = 3
STOP_DURATION = 1.0  # seconds
COOLDOWN_DURATION = 5.0  # seconds

# Stop detection state
consecutive_stop_count = 0
stop_start_time = None
ignore_stop_until = None

# stop the robot 
bot.setVelocity(0, 0)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 5)

        self.relu = nn.ReLU()


    def forward(self, x):
        #extract features with convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        #linear layer for classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
       
        return x
    
net = Net()
stop_sign_detector = StopSignDetector()

#LOAD NETWORK WEIGHTS HERE
net.load_state_dict(torch.load('steer_net.pth'))

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

controller.stopped = True

try:
    angle = 0
    while True:
        # get an image from the the robot
        current_time = time.time()
        im = bot.getImage()

        # Check if we're in cooldown period
        in_cooldown = (ignore_stop_until is not None and current_time < ignore_stop_until)

        # TO DO: apply any necessary image transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((40, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
        img = transform(im)

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
            stop_detected = stop_sign_detector.detect(im)
            
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

        # Pass image through network get a prediction
        outputs = net(img.unsqueeze(0)) # add batch dimension
        _, prediction = torch.max(outputs, 1)

        # Convert prediction into a meaningful steering angle
        # angle = LABEL_TO_ANGLE[prediction.item()]

        left,right = controller(prediction.item())
            
        bot.setVelocity(left, right)
        # bot.setVelocity(0, 0)
        print(f"Predicted class: {prediction.item()}, controller_angle: {controller.angle:.2f}, left: {left}, right: {right}")
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
