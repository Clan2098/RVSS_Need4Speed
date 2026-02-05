import cv2
import numpy as np

class HSVTuner:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        height = self.image.shape[0]
        self.image = self.image[height//2:, :]  # Crop like your detector
        
        # Starting values (conservative)
        self.h_min1 = 0
        self.h_max1 = 5
        self.s_min = 100
        self.v_min = 130
        self.h_min2 = 160
        self.h_max2 = 180
        
    def nothing(self, x):
        pass
    
    def tune(self):
        cv2.namedWindow('HSV Tuner')
        
        # Create trackbars
        cv2.createTrackbar('H Min (Red1)', 'HSV Tuner', self.h_min1, 20, self.nothing)
        cv2.createTrackbar('H Max (Red1)', 'HSV Tuner', self.h_max1, 20, self.nothing)
        cv2.createTrackbar('S Min', 'HSV Tuner', self.s_min, 255, self.nothing)
        cv2.createTrackbar('V Min', 'HSV Tuner', self.v_min, 255, self.nothing)
        cv2.createTrackbar('H Min (Red2)', 'HSV Tuner', self.h_min2, 180, self.nothing)
        cv2.createTrackbar('H Max (Red2)', 'HSV Tuner', self.h_max2, 180, self.nothing)
        
        print("\nControls:")
        print("- Adjust trackbars to tune HSV ranges")
        print("- Click on image to see pixel values")
        print("- Press 'q' to quit")
        print("- Press 's' to save current settings")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                bgr = self.image[y, x]
                hsv_pixel = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)[y, x]
                print(f"\n📍 Clicked at ({x}, {y}):")
                print(f"   HSV: H={hsv_pixel[0]:3d}, S={hsv_pixel[1]:3d}, V={hsv_pixel[2]:3d}")
                print(f"   RGB: R={bgr[2]:3d}, G={bgr[1]:3d}, B={bgr[0]:3d}")
        
        cv2.setMouseCallback('HSV Tuner', mouse_callback)
        
        while True:
            # Get current trackbar positions
            h_min1 = cv2.getTrackbarPos('H Min (Red1)', 'HSV Tuner')
            h_max1 = cv2.getTrackbarPos('H Max (Red1)', 'HSV Tuner')
            s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
            v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')
            h_min2 = cv2.getTrackbarPos('H Min (Red2)', 'HSV Tuner')
            h_max2 = cv2.getTrackbarPos('H Max (Red2)', 'HSV Tuner')
            
            # Convert to HSV
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            # Create masks
            lower_red1 = np.array([h_min1, s_min, v_min])
            upper_red1 = np.array([h_max1, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            
            lower_red2 = np.array([h_min2, s_min, v_min])
            upper_red2 = np.array([h_max2, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply result as overlay
            result = self.image.copy()
            result[mask > 0] = [0, 255, 0]  # Green overlay on detected areas
            
            # Create display with multiple views
            hsv_display = np.zeros_like(self.image)
            hsv_display[:,:,0] = hsv[:,:,0]  # Show hue
            hsv_display[:,:,1] = hsv[:,:,1]  # Show saturation
            hsv_display[:,:,2] = hsv[:,:,2]  # Show value
            
            top_row = np.hstack([self.image, result])
            bottom_row = np.hstack([
                cv2.cvtColor(hsv[:,:,1], cv2.COLOR_GRAY2BGR),  # Saturation
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)          # Final mask
            ])
            
            display = np.vstack([top_row, bottom_row])
            
            # Add text with current values
            text = f"Red1: H={h_min1}-{h_max1} | Red2: H={h_min2}-{h_max2} | S>={s_min} | V>={v_min}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            pixel_count = np.sum(mask > 0)
            cv2.putText(display, f"Detected pixels: {pixel_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('HSV Tuner', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n" + "="*60)
                print("SAVED SETTINGS:")
                print("="*60)
                print(f"self.lower_red1 = np.array([{h_min1}, {s_min}, {v_min}])")
                print(f"self.upper_red1 = np.array([{h_max1}, 255, 255])")
                print(f"self.lower_red2 = np.array([{h_min2}, {s_min}, {v_min}])")
                print(f"self.upper_red2 = np.array([{h_max2}, 255, 255])")
                print("="*60)
        
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Test with one of your images
    tuner = HSVTuner("/home/arnaud/Documents/ETH/Master/Thesis/RVSS/RVSS_Need4Speed/data/train_stop_signs/stop/0001360.0_2.jpg")
    tuner.tune()