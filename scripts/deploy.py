#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
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
net.load_state_dict(torch.load('working_845.pth'))

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

try:
    angle = 0
    while True:
        # get an image from the the robot
        im = bot.getImage()

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

        #TO DO: check for stop signs?
        
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
