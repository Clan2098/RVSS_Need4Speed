import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path

class StopDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder, "**", "*" + self.img_ext), recursive=True)            
        self.totensor = transforms.ToTensor()
        self.class_labels = ['free', 'stop']
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)[120:, :, :] # Crop the image to remove the car hood: only keep the bottom half
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Conert to HSV color space
        
        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        # Extract label from parent folder name
        folder_name = path.basename(path.dirname(f))
        label = self.class_labels.index(folder_name)  # 0 for 'free', 1 for 'stop'
                      
        return img, label