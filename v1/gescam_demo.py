#!/usr/bin/env python3
"""
GESCAM Demo with Face Detection
This demo shows how to use the GESCAM model for gaze prediction
with a real face detector for head tracking.
"""

import cv2
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from v1.gescam_inference import GazeInference

def load_face_detector():
    """Load the OpenCV face detector"""
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        # Try alternative path for Colab/Kaggle
        alt_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        if os.path.exists(alt_path):
            face_cascade = cv2.CascadeClassifier(alt_path)
        else:
            raise RuntimeError("Error: Could not load face cascade classifier")
    
    return face_cascade

def face_tracker(frame, face_cascade):
    """Track faces in the frame"""
    def tracker(frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convert to normalized coordinates
        height, width = frame.shape[:2]
        head_boxes = []
        
        for (x, y, w, h) in faces:
            # Convert to normalized coordinates
            x1, y1 = x / width, y / height
            x2, y2 = (x + w) / width, (y + h) / height
            
            head_boxes.append([x1, y1, x2, y2])
        
        return head_boxes
    
    return tracker

def demo_image(model_path, image_path, output_dir):
    """Run demo on a single image"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load face detector
    face_cascade = load_face_detector()
    
    # Initialize inference model
    inference = GazeInference(model_path)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Error loading image: {image_path}")
    
    # Convert to RGB for face detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    tracker = face_tracker(img_rgb, face_cascade)
    head_boxes = tracker(img_rgb)
    
    if not head_boxes:
        print("No faces detected in the image")
        # Use a default box for demonstration
        height, width = img.shape[:2]
        head_boxes = [[0.4, 0.4, 0.6, 0.6]]
    
    # Convert to PIL for inference
    img_pil = Image.fromarray(img_rgb)
    
    # Process each detected face
    for i, head_bbox in enumerate(head_boxes):
        # Predict gaze
        gaze_heatmap, in_frame_prob, visualization = inference.predict(
            img_pil, head_bbox
        )
        
        # Save visualization
        vis_path = os.path.join(output_dir, f'gaze_face_{i}.png')
        cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGBA2BGRA))
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, f'heatmap_face_{i}.png')
        plt.imsave(heatmap_path, gaze_heatmap, cmap='jet')
    
    print(f"Processed {len(head_boxes)} faces. Results saved to {output_dir}")

def demo_video(model_path, video_path, output_dir):
    """Run demo on a video"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load face detector
    face_cascade = load_face_detector()
    
    # Initialize inference model
    inference = GazeInference(model_path)
    
    # Create face tracker
    tracker = face_tracker(None, face_cascade)
    
    # Output video path
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"gaze_{video_name}")
    
    # Process video
    inference.process_video(
        video_path,
        output_path=output_path,
        head_tracker=tracker,
        detector=None,  # No object detector for simplicity
        sample_rate=5  # Process every 5th frame for speed
    )
    
    print(f"Video processing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='GESCAM Demo with Face Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--output', type=str, default='./output', help='Path to output directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.video:
        raise ValueError("Either --image or --video must be provided")
    
    # Run demo on image
    if args.image:
        demo_image(args.model, args.image, args.output)
    
    # Run demo on video
    if args.video:
        demo_video(args.model, args.video, args.output)

if __name__ == "__main__":
    main()