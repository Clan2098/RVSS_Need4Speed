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
from steer_labels import LABELS, steering_to_class, label_to_class, LABEL_TO_ANGLE
from controller import Controller

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

controller = Controller()

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

#LOAD NETWORK WEIGHTS HERE
net.load_state_dict(torch.load('steer_net.pth'))

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

        #TO DO: apply any necessary image transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((40, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
        img = transform(im) 



        #TO DO: pass image through network get a prediction
        outputs = net(img.unsqueeze(0)) # add batch dimension
        _, prediction = torch.max(outputs, 1)

        #TO DO: convert prediction into a meaningful steering angle
        # angle = LABEL_TO_ANGLE[prediction.item()]

        left,right = controller(prediction.item())

        #TODO: convert prediction into a meaningful steering angle
        
        # angle = 0

        # Kd = 20 #base wheel speeds, increase to go faster, decrease to go slower
        # Ka = 20 #how fast to turn when given an angle
        # left  = int(Kd + Ka*angle)
        # right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
        #bot.setVelocity(0, 0)
        print(f"Predicted class: {prediction.item()}, controller_angle: {controller.angle:.2f}, left: {left}, right: {right}")
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
