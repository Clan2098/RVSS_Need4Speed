import numpy as np
import cv2
import math
from machinevisiontoolbox import Image

class StopSignDetector:
    """Lightweight stop sign detector using blob grouping."""

    def __init__(self):
        # HSV color ranges for red
        self.lower_red1 = np.array([0, 100, 130])
        self.upper_red1 = np.array([5, 255, 255])
        self.lower_red2 = np.array([160, 100, 130])
        self.upper_red2 = np.array([180, 255, 255])

        # Grouping and validation parameters
        self.max_blob_distance = 30
        self.min_bbox_width = 20
        self.max_bbox_width = 300
        self.min_bbox_height = 20
        self.max_bbox_height = 300
    
    def calculate_distance(self, centroid1, centroid2):
        x1, y1 = centroid1
        x2, y2 = centroid2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def group_nearby_blobs(self, blobs):
        if len(blobs) == 0:
            return []
        
        groups = [[blob] for blob in blobs]
        merged = True
        
        while merged:
            merged = False
            new_groups = []
            used = set()
            
            for i, group1 in enumerate(groups):
                if i in used:
                    continue
                
                for j, group2 in enumerate(groups[i+1:], start=i+1):
                    if j in used:
                        continue
                    
                    should_merge = False
                    for blob1 in group1:
                        for blob2 in group2:
                            if self.calculate_distance(blob1.centroid, blob2.centroid) < self.max_blob_distance:
                                should_merge = True
                                break
                        if should_merge:
                            break
                    
                    if should_merge:
                        group1.extend(group2)
                        used.add(j)
                        merged = True
                
                new_groups.append(group1)
                used.add(i)
            
            groups = new_groups
        
        return groups
    
    def get_group_bounding_box(self, blob_group):
        all_bboxes = [blob.bbox for blob in blob_group]
        
        x_min = min(bbox[0] for bbox in all_bboxes)
        x_max = max(bbox[1] for bbox in all_bboxes)
        y_min = min(bbox[2] for bbox in all_bboxes)
        y_max = max(bbox[3] for bbox in all_bboxes)
        
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        return x_min, x_max, y_min, y_max, width, height, center_x, center_y
    
    def is_valid_stop_sign_group(self, blob_group):
        x_min, x_max, y_min, y_max, width, height, cx, cy = self.get_group_bounding_box(blob_group)
        
        if width < self.min_bbox_width and height < self.min_bbox_height:
            return False
        
        return True
    
    def detect(self, image):
        """
        Detect stop signs in the image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            bool: True if stop sign detected, False otherwise
        """
        height, _ = image.shape[:2]
        image_cropped = image[height//2:, :]
        
        hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if not np.any(mask):
            return False
        
        try:
            blobs = Image(mask).blobs()
        except (IndexError, ValueError):
            return False
        
        groups = self.group_nearby_blobs(blobs)
        
        for group in groups:
            if self.is_valid_stop_sign_group(group):
                return True
        
        return False