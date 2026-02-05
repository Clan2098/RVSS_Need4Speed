import numpy as np
import cv2
import math
import glob
from machinevisiontoolbox import Image, Kernel
import matplotlib.pyplot as plt

class stop_sign_detector:
    """
    A simple stop sign detector using blob detector from machinevisiontoolbox.
    """

    def __init__(self):
        # Define the red color range in HSV
        self.lower_red1 = np.array([0, 100, 130])
        self.upper_red1 = np.array([5, 255, 255])
        self.lower_red2 = np.array([160, 100, 130])
        self.upper_red2 = np.array([180, 255, 255])

        self.max_blob_distance = 30
        self.min_bbox_width = 20     # Minimum bounding box width
        self.max_bbox_width = 300    # Maximum bounding box width
        self.min_bbox_height = 20    # Minimum bounding box height
        self.max_bbox_height = 300   # Maximum bounding box height
    
    def calculate_distance(self, centroid1, centroid2):
        """Calculate Euclidean distance between two centroids"""
        x1, y1 = centroid1
        x2, y2 = centroid2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def group_nearby_blobs(self, blobs):
        """
        Group blobs whose centroids are close together.
        Returns list of blob groups (each group is a list of blobs).
        """
        if len(blobs) == 0:
            return []
        
        # Start with each blob in its own group
        groups = [[blob] for blob in blobs]
        
        # Iteratively merge groups whose blobs are close
        merged = True
        while merged:
            merged = False
            new_groups = []
            used = set()
            
            for i, group1 in enumerate(groups):
                if i in used:
                    continue
                
                # Check if this group should merge with any other
                for j, group2 in enumerate(groups[i+1:], start=i+1):
                    if j in used:
                        continue
                    
                    # Check if any blob in group1 is close to any blob in group2
                    should_merge = False
                    for blob1 in group1:
                        for blob2 in group2:
                            dist = self.calculate_distance(blob1.centroid, blob2.centroid)
                            if dist < self.max_blob_distance:
                                should_merge = True
                                break
                        if should_merge:
                            break
                    
                    if should_merge:
                        # Merge the groups
                        group1.extend(group2)
                        used.add(j)
                        merged = True
                
                new_groups.append(group1)
                used.add(i)
            
            groups = new_groups
        
        return groups
    
    def get_group_bounding_box(self, blob_group):
        """
        Calculate bounding box that encompasses all blobs in a group.
        Returns (x_min, x_max, y_min, y_max, width, height, center_x, center_y)
        """
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
        """
        Check if a group of blobs looks like a stop sign based on bounding box size.
        """
        x_min, x_max, y_min, y_max, width, height, cx, cy = self.get_group_bounding_box(blob_group)
        
        # Check bounding box dimensions
        if width < self.min_bbox_width and height < self.min_bbox_height:
            return False, f"width {width:.0f} and height {height:.0f} not in [{self.min_bbox_width}, {self.max_bbox_width}] and [{self.min_bbox_height}, {self.max_bbox_height}]"
        
        # if height < self.min_bbox_height or height > self.max_bbox_height:
        #     return False, f"height {height:.0f} not in [{self.min_bbox_height}, {self.max_bbox_height}]"
        
        return True, f"valid: {len(blob_group)} blobs, {width:.0f}x{height:.0f}"
    
    def detect_with_blobs(self, image, display_raw=False):
        """
        Detect stop signs in the given image.
        Parameters:
            image (numpy.ndarray): The input image in BGR format from OpenCV.
        """
        # Crop to bottom half only
        height, _ = image.shape[:2]
        image = image[height//2:, :]  # Remove top half
        crop_offset_y = height // 2
        
        # Convert BGR to HSV (OpenCV operations)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert to machinevisiontoolbox Image only for blob detection
        mv_mask = Image(mask)

        if np.any(mask):
            try:
                blobs = mv_mask.blobs()
                print(f"Detected {len(blobs)} initial blobs")
            except (IndexError, ValueError) as e:
                print(f"Warning: Blob detection failed ({type(e).__name__})")
                return []
        else:
            print("No red regions detected in mask")
            return []
        
        # Group nearby blobs
        groups = self.group_nearby_blobs(blobs)
        print(f"Formed {len(groups)} blob groups")
        
        # Validate each group
        detected_signs = []
        valid_groups = []
        
        for i, group in enumerate(groups):
            is_valid, reason = self.is_valid_stop_sign_group(group)
            
            if is_valid:
                x_min, x_max, y_min, y_max, width, height, cx, cy = self.get_group_bounding_box(group)
                
                # Adjust y-coordinate for original image
                cy_original = cy + crop_offset_y
                
                print(f"  Group {i+1}: ✓ {reason}")
                
                detected_signs.append({
                    'center': (int(cx), int(cy_original)),
                    'bbox': (int(x_min), int(x_max), int(y_min), int(y_max)),
                    'width': int(width),
                    'height': int(height),
                    'num_blobs': len(group)
                })
                valid_groups.append((group, x_min, x_max, y_min, y_max, width, height))
            else:
                print(f"  Group {i+1}: ✗ {reason}")
        
        # Display if requested
        if display_raw:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Original Image - {len(blobs)} blobs detected')
            axes[0].axis('off')
            
            # Mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Red Mask')
            axes[1].axis('off')
            
            # Detected stop signs with bounding boxes
            result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
            
            # Draw all blob centroids in blue (for debugging)
            for blob in blobs:
                cx, cy = blob.centroid
                cv2.circle(result, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            
            # Draw valid groups with green bounding boxes
            for group, x_min, x_max, y_min, y_max, width, height in valid_groups:
                # Draw bounding box
                cv2.rectangle(result, (int(x_min), int(y_min)), (int(x_max), int(y_max)), 
                            (0, 255, 0), 2)
                
                # Draw center
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                cv2.circle(result, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                
                # Add size label
                cv2.putText(result, f'{int(width)}x{int(height)}', 
                          (int(x_min), int(y_min) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            axes[2].imshow(result)
            axes[2].set_title(f'Detected Stop Signs: {len(valid_groups)}')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            plt.close('all')
    

        # if np.any(mask):
        #     # Detect blobs in the mask
        #     blobs = mv_mask.blobs()
        # else:
        #     blobs = []
        #     print("No red regions detected in mask")
        
        # valid_blobs = []
        # detected_signs = []

        # for blob in blobs:
        #     bbox = blob.bbox  # Returns (umin, umax, vmin, vmax)
        #     width = bbox[1] - bbox[0]
        #     height = bbox[3] - bbox[2]
        #     aspect_ratio = width / height if height > 0 else 0

            # # Filter by bounding box size
            # if width < 30 or width > 300:  # Adjust based on your camera
            #     print(f"Rejected blob: width {width} out of range")
            #     continue
            
            # if height < 30 or height > 300:
            #     print(f"Rejected blob: height {height} out of range")
            #     continue
            
            # # Stop signs on the floor should have reasonable aspect ratio
            # # Even with perspective, they shouldn't be too elongated
            # if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            #     print(f"Rejected blob: aspect ratio {aspect_ratio:.2f}")
            #     continue
            
            # # Filter by total area (combined area of the merged blob)
            # if blob.area < 500 or blob.area > 20000:  # Adjust thresholds
            #     print(f"Rejected blob: area {blob.area}")
            #     continue

        #     x, y = blob.centroid
        #     print(f"Detected stop sign at ({x}, {y + height//2}) with bbox {width}x{height}")
        #     detected_signs.append((int(x), int(y + height//2), width, height))
        #     valid_blobs.append(blob)

        # # Display if requested
        # if display_raw:
        #     # Convert BGR to RGB for proper display
        #     mv_original = Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     mv_original.disp(title="Original Image (cropped)", block=False)
            
        #     Image(mask).disp(title="Red mask after morphology", block=False)
            
        #     if len(blobs) > 0:
        #         print(f"Detected {len(blobs)} blobs total, {len(detected_signs)} after filtering.")
        #         # Display the image FIRST, then plot blobs on it
        #         mv_original.disp(title="Detected Signs", block=False)
        #         for blob in valid_blobs:
        #             blob.plot_labelbox(color="red", linewidth=2)
        
        #     plt.show()

        # plt.close('all') 
        
        return detected_signs
    
    def detect(self, image, display_raw=False):
        return self.detect_with_blobs(image, display_raw)
    
if __name__ == "__main__":
    detector = stop_sign_detector()
    image_folder = "/home/arnaud/Documents/ETH/Master/Thesis/RVSS/RVSS_Need4Speed/data/train_stop_signs/stop/"
    image_paths = glob.glob(f"{image_folder}/*.jpg")

    # for img_path in image_paths:
    #     print(img_path)
    #     test_image = cv2.imread(img_path)
    #     signs = detector.detect(test_image, display_raw=True)
    #     print(f"Image: {img_path} - Detected stop signs (x, y, radius):", signs)

    for img_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path}")
        print('='*60)
        test_image = cv2.imread(img_path)
        signs = detector.detect(test_image, display_raw=True)
        
        print(f"\n✓ Found {len(signs)} stop sign(s)")
        for i, sign in enumerate(signs):
            print(f"  Sign {i+1}: center={sign['center']}, size={sign['width']}×{sign['height']}, blobs={sign['num_blobs']}")