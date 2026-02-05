import numpy as np
import cv2
import math
import glob
from machinevisiontoolbox import Image, Kernel
import matplotlib.pyplot as plt
from ultralytics import YOLO


class stop_sign_detector:
    """
    A simple stop sign detector using blob detector from machinevisiontoolbox.
    """

    def __init__(self, use_yolo=False, yolo_model_path='yolov8n.pt'):
        # Define the red color range in HSV
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        self.lower_white = np.array([0, 0, 200])      # Any hue, low saturation, high brightness
        self.upper_white = np.array([180, 50, 255]) 

        # YOLO setup
        self.use_yolo = use_yolo
        if use_yolo:
            self.yolo_model = YOLO(yolo_model_path)

    def detect_with_yolo(self, image, display_raw=False):
        """
        Detect stop signs in the given image using YOLOv8.
        Parameters:
            image (numpy.ndarray): The input image in BGR format from OpenCV.
        """
        # Crop to bottom half only
        height, _ = image.shape[:2]
        image = image[height//2:, :]  # Remove top half

        results = self.yolo_model(image)

        detected_signs = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detected_signs.append((cx, cy))
                print(f"Detected stop sign at ({cx}, {cy}) using YOLO")

        # Display if requested
        if display_raw:
            # YOLO has built-in visualization
            annotated_image = results[0].plot()
            
            # Convert to RGB for display
            mv_image = Image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            mv_image.disp(title="YOLO Detection", block=False)

            plt.show()
            plt.close('all')

        return detected_signs
    
    def detect_with_blobs(self, image, display_raw=False):
        """
        Detect stop signs in the given image.
        Parameters:
            image (numpy.ndarray): The input image in BGR format from OpenCV.
        """
        # Crop to bottom half only
        height, _ = image.shape[:2]
        image = image[height//2:, :]  # Remove top half
        
        # Convert BGR to HSV (OpenCV operations)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Convert to machinevisiontoolbox Image only for blob detection
        mv_mask = Image(mask)
        mv_white_mask = Image(mask)

        if np.any(mask):
            # Detect blobs in the mask
            blobs = mv_white_mask.blobs()
        else:
            blobs = []
            print("No red regions detected in mask")
        
        valid_blobs = []
        detected_signs = []
        for blob in blobs:
            # Filter by area
            if blob.area < 50 or blob.area > 10000:
                print(f"Rejected blob with area: {blob.area}")
                continue
            
            # # Filter by circularity (octagons are ~0.85)
            # if blob.circularity < 0.70 or blob.circularity > 0.95:
            #     print(f"Rejected blob with circularity: {blob.circularity}")
            #     continue
            
            x, y = blob.centroid
            print(f"Detected stop sign at ({x}, {y}) with area {blob.area}")
            detected_signs.append((int(x), int(y)))
            valid_blobs.append(blob)
        # Display if requested
        if display_raw:
            # Convert BGR to RGB for proper display
            mv_original = Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            mv_original.disp(title="Original Image", block=False)
            
            Image(mask).disp(title="Red mask", block=False)
            
            if len(blobs) > 0:
                print(f"Detected {len(blobs)} blobs total, {len(detected_signs)} after filtering.")
                # Display the image FIRST, then plot blobs on it
                mv_original.disp(title="Detected Signs", block=False)
                for blob in valid_blobs:
                    blob.plot_labelbox(color="red", linewidth=2)
        
        plt.show()
        plt.close('all') 
        
        return detected_signs
    
    def detect(self, image, display_raw=False):
        if self.use_yolo:
            return self.detect_with_yolo(image, display_raw)
        else:
            return self.detect_with_blobs(image, display_raw)
    
if __name__ == "__main__":
    detector = stop_sign_detector(use_yolo=False, yolo_model_path='yolov8n.pt')
    image_folder = "/home/arnaud/Documents/ETH/Master/Thesis/RVSS/RVSS_Need4Speed/data/train_stop/"
    image_paths = glob.glob(f"{image_folder}/*.jpg")

    for img_path in image_paths:
        test_image = cv2.imread(img_path)
        signs = detector.detect(test_image, display_raw=True)
        print(f"Image: {img_path} - Detected stop signs (x, y, radius):", signs)