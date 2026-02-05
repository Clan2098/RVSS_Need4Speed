import cv2
import os
from pathlib import Path
from glob import glob

def flip_angle(angle_str):
    """
    Negate the angle value (string representation).
    Example: "-0.3" becomes "0.3", "0.5" becomes "-0.5"
    """
    try:
        angle = float(angle_str)
        return str(-angle)
    except ValueError:
        return angle_str

def flip_images(input_folder, output_folder, img_ext=".jpg"):
    """
    Recursively find all images in input folder and subfolders,
    flip them horizontally, negate the angle in filename, and save to output folder 
    while preserving directory structure.
    
    Args:
        input_folder: Path to the input folder containing images
        output_folder: Path where flipped images will be saved
        img_ext: Image file extension to search for (default: ".jpg")
    """
    # Convert to Path objects for easier manipulation
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images recursively
    image_files = glob(os.path.join(input_folder, "**", f"*{img_ext}"), recursive=True)
    
    if not image_files:
        print(f"No images with extension '{img_ext}' found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    for image_path in image_files:
        # Read the image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Could not read {image_path}")
            continue
        
        # Flip the image horizontally
        flipped_img = cv2.flip(img, 1)
        
        # Preserve directory structure in output folder
        relative_path = Path(image_path).relative_to(input_path)
        
        # Extract filename and extension
        filename = relative_path.name
        name_without_ext = filename[:-len(img_ext)]
        
        # Extract angle from filename (7th character onwards, index 6)
        if len(name_without_ext) > 6:
            prefix = name_without_ext[:6]  # First 6 characters
            angle_str = name_without_ext[6:]  # Angle starts at index 6
            flipped_angle = flip_angle(angle_str)
            new_filename = f"{prefix}{flipped_angle}{img_ext}"
        else:
            new_filename = filename
        
        output_image_path = output_path / relative_path.parent / new_filename
        
        # Create subdirectories if needed
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the flipped image
        cv2.imwrite(str(output_image_path), flipped_img)
        print(f"Processed: {relative_path} -> {new_filename}")
    
    print(f"Done! Flipped images saved to {output_folder}")


if __name__ == "__main__":
    # Example usage
    input_dir = "train"
    output_dir = "train_flipped"
    
    flip_images(input_dir, output_dir, img_ext=".jpg")

