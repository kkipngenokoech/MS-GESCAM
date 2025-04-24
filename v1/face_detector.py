#!/usr/bin/env python3
"""
DNN-based Face Detector Test Script (Fixed)
This script uses OpenCV's DNN face detector which is more robust than the cascade classifier.
"""

import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import sys

def detect_faces_dnn(image_path, confidence_threshold=0.5):
    """
    Detect faces using DNN-based detector
    
    Args:
        image_path: Path to image
        confidence_threshold: Minimum probability to consider a detection valid
        
    Returns:
        faces: List of detected faces (x1, y1, x2, y2)
        img: Original image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Load the DNN model - using OpenCV's built-in face detector
    # This avoids the need to download external model files
    try:
        # Try using OpenCV's built-in DNN face detector
        net = cv2.dnn.readNetFromCaffe(
            cv2.data.haarcascades + '../dnn/deploy.prototxt',
            cv2.data.haarcascades + '../dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        )
    except:
        print("Could not load built-in OpenCV DNN model. Checking for local files...")
        
        # Check if files exist locally
        prototxt = "deploy.prototxt"
        model = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if os.path.exists(prototxt) and os.path.exists(model):
            print("Using local model files.")
            net = cv2.dnn.readNetFromCaffe(prototxt, model)
        else:
            print("DNN model files not found. Please install the full OpenCV package or manually download:")
            print("1. deploy.prototxt")
            print("2. res10_300x300_ssd_iter_140000.caffemodel or res10_300x300_ssd_iter_140000_fp16.caffemodel")
            print("\nAlternatively, use the MediaPipe face detector (recommended).")
            return [], img_rgb
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), 
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    
    # Set the input to the network
    net.setInput(blob)
    
    # Forward pass to get detections
    detections = net.forward()
    
    # Collect faces
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter by confidence
        if confidence > confidence_threshold:
            # Get face coordinates (0-1 range)
            box = detections[0, 0, i, 3:7]
            
            # Scale to image size
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(x1+1, min(x2, width))
            y2 = max(y1+1, min(y2, height))
            
            faces.append((x1, y1, x2, y2))
    
    return faces, img_rgb

def detect_faces_mediapipe(image_path):
    """
    Detect faces using MediaPipe Face Detection
    
    Args:
        image_path: Path to image
        
    Returns:
        faces: List of detected faces (x1, y1, x2, y2)
        img: Original image
    """
    try:
        import mediapipe as mp
    except ImportError:
        print("MediaPipe not installed. Install with: pip install mediapipe")
        return [], None
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    
    # Use face detection model
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Process the image
        results = face_detection.process(img_rgb)
        
        # Collect faces
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                box = detection.location_data.relative_bounding_box
                
                # Convert to pixel coordinates
                x1 = int(box.xmin * width)
                y1 = int(box.ymin * height)
                x2 = int((box.xmin + box.width) * width)
                y2 = int((box.ymin + box.height) * height)
                
                # Ensure coordinates are within image boundaries
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(x1+1, min(x2, width))
                y2 = max(y1+1, min(y2, height))
                
                faces.append((x1, y1, x2, y2))
    
    return faces, img_rgb

def visualize_faces(img, faces):
    """Visualize detected faces on the image"""
    # Create a copy of the image
    img_copy = img.copy()
    
    # Draw rectangles around detected faces
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img_copy

def main():
    parser = argparse.ArgumentParser(description='Advanced Face Detection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output image')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold (lower values may detect more faces)')
    parser.add_argument('--method', type=str, default='mediapipe', choices=['dnn', 'mediapipe'],
                       help='Face detection method to use')
    
    args = parser.parse_args()
    
    # Detect faces
    try:
        if args.method == 'dnn':
            faces, img = detect_faces_dnn(args.image, confidence_threshold=args.confidence)
        else:
            faces, img = detect_faces_mediapipe(args.image)
            
        if img is None:
            print("Failed to process image.")
            return
        
        # Visualize faces
        img_with_faces = visualize_faces(img, faces)
        
        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(img_with_faces)
        plt.title(f"Detected {len(faces)} faces using {args.method}")
        plt.axis('off')
        
        # Print face coordinates
        print(f"Detected {len(faces)} faces:")
        for i, (x1, y1, x2, y2) in enumerate(faces):
            print(f"Face {i+1}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            # Calculate normalized coordinates (for use with GESCAM)
            height, width = img.shape[:2]
            x1_norm, y1_norm = x1/width, y1/height
            x2_norm, y2_norm = x2/width, y2/height
            print(f"  Normalized: [{x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}]")
        
        # Save result if output path is specified
        if args.output:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            plt.savefig(args.output, bbox_inches='tight')
            print(f"Result saved to {args.output}")
        
        # Only try to show if we're in an interactive environment
        if hasattr(sys, 'ps1') or sys.flags.interactive:
            plt.show()
        else:
            plt.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()