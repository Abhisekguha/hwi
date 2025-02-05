import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
from torchvision.ops import nms
import os
from pathlib import Path
import glob

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

def load_and_preprocess_image(image_path):
    """
    Load and preprocess TIF image with debug logging
    """
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image")
    print(f"Image loaded successfully. Shape: {image.shape}")
    return image

def sliding_window(image, window_size=(1000, 1000), overlap=100):
    """
    Generate sliding window coordinates with proper overlap handling
    """
    windows = []
    h, w = image.shape[:2]
    print(f"Original image size: {w}x{h}")
    
    # Calculate stride (step size)
    stride_h = window_size[0] - overlap
    stride_w = window_size[1] - overlap
    
    # Generate windows only for the valid width
    for y in range(0, h, stride_h):
        # Only create one window for the width since it's smaller than window_size
        window_h = min(window_size[0], h - y)
        window_w = w  # Use full width
        
        windows.append((0, y, window_w, y + window_h))
        print(f"Added window: (0, {y}, {window_w}, {y + window_h})")
    
    print(f"Generated {len(windows)} windows")
    return windows

def detect_in_window(model, image, window, conf_thres=0.25):
    """
    Perform detection on a single window with proper score handling
    """
    x1, y1, x2, y2 = window
    window_img = image[y1:y2, x1:x2]
    
    print(f"Processing window: {window}")
    print(f"Window shape: {window_img.shape}")
    
    # Resize window to model input size
    target_size = (1000, 1000)
    if window_img.shape[:2] != target_size:
        window_img = cv2.resize(window_img, target_size)
        print(f"Resized window to: {window_img.shape}")
    
    # Convert to PIL Image
    window_img_pil = Image.fromarray(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))
    
    # Perform detection with confidence threshold
    results = model(window_img_pil, conf=conf_thres)
    
    boxes = []
    scores = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # Convert boxes to numpy for manipulation
            boxes_data = result.boxes.xyxy.cpu().numpy()
            scores_data = result.boxes.conf.cpu().numpy()
            
            # Scale coordinates back to original window size
            if window_img.shape[:2] != (y2-y1, x2-x1):
                scale_x = (x2 - x1) / target_size[0]
                scale_y = (y2 - y1) / target_size[1]
                
                boxes_data[:, [0, 2]] *= scale_x
                boxes_data[:, [1, 3]] *= scale_y
            
            # Adjust coordinates to original image space
            boxes_data[:, [0, 2]] += x1
            boxes_data[:, [1, 3]] += y1
            
            boxes.extend(boxes_data.tolist())
            scores.extend(scores_data.tolist())
    
    print(f"Found {len(boxes)} detections in this window")
    return boxes, scores

def process_single_image(image_path, model, output_path, conf_thres=0.25):
    """
    Process a single image with the model
    """
    try:
        # Load and preprocess image
        image = load_and_preprocess_image(image_path)
        
        # Generate sliding windows
        windows = sliding_window(image, window_size=(1000, 1000), overlap=100)
        
        # Collect all detections
        all_boxes = []
        all_scores = []
        
        # Perform detection on each window
        for i, window in enumerate(windows):
            print(f"\nProcessing window {i+1}/{len(windows)}")
            boxes, scores = detect_in_window(model, image, window, conf_thres)
            
            if boxes and scores:  # Only add if detections were found
                all_boxes.extend(boxes)
                all_scores.extend(scores)
        
        print(f"\nTotal detections before NMS: {len(all_boxes)}")
        
        # Perform NMS if detections were found
        if all_boxes:
            # Convert to tensors, ensuring proper formatting
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
            
            # Perform NMS
            nms_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.45)
            print(f"Detections after NMS: {len(nms_indices)}")
            
            # Draw final boxes
            output_image = image.copy()
            for idx in nms_indices:
                box = boxes_tensor[idx].numpy().astype(int)
                score = scores_tensor[idx].item()
                
                # Draw rectangle
                cv2.rectangle(output_image, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (0, 255, 0), 2)
                
                # Add confidence score
                cv2.putText(output_image, 
                           f"{score:.2f}", 
                           (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (0, 255, 0), 
                           2)
            
            # Save output image
            cv2.imwrite(output_path, output_image)
            print(f"Output saved to {output_path}")
            return True
        else:
            print("No detections found after processing all windows.")
            return False
            
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return False


def process_folder(input_folder, output_folder, model_path, conf_thres=0.25):
    """
    Process all TIF images in a folder and save results to output folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model once for all images
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully")
    
    # Get all TIF files in the input folder
    tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    print(f"Found {len(tif_files)} TIF files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, tif_file in enumerate(tif_files):
        print(f"\nProcessing image: {tif_file}")
        
        # Extract input file name without extension
        input_filename = Path(tif_file).stem
        
        # Generate output path with input file name and order number
        order_number = i + 1
        output_path = os.path.join(output_folder, f"{input_filename}-Inf_{order_number}.png")
        
        # Process the image
        if process_single_image(tif_file, model, output_path, conf_thres):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
    print(f"Output images saved to: {output_folder}")

if __name__ == "__main__":
    # Define your folders and model path
    input_folder = r"C:\Users\ADMIN\Desktop\Detection\Original-Handwritten Text-Dataset-Jan-2025"
    output_folder = r"C:\Users\ADMIN\Desktop\Detection\inference_models_testing\Inference2-HTD-Jan2025-yolov10n-ep300-bt10"
    model_path = r"C:\Users\ADMIN\Desktop\Detection\models\detect_yolov10n_ep300_bt_10\train\weights\best.pt"
    
    # Process all images in the folder
    process_folder(input_folder, output_folder, model_path, conf_thres=0.25)
