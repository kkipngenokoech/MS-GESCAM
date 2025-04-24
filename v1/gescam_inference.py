import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from v1.modelArch import MSGESCAMModel


class GazeInference:
    def __init__(self, model_path, device=None):
        """
        Initialize the gaze inference module
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (None for auto-detection)
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = MSGESCAMModel(pretrained=False, output_size=64)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded model weights")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
        # Set model to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Define tensor to PIL conversion for visualization
        self.to_pil = transforms.ToPILImage()
        
        print("GESCAM inference model initialized successfully")
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            scene_img: Preprocessed scene image tensor
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                # Assume it's already a PIL image or numpy array
                if isinstance(image_path, np.ndarray):
                    image = Image.fromarray(image_path)
                else:
                    image = image_path
            
            # Apply transforms
            scene_img = self.transform(image)
            
            return scene_img, image
        except Exception as e:
            raise RuntimeError(f"Error preprocessing image: {e}")
    
    def create_head_position_mask(self, head_bbox, img_size=(224, 224)):
        """
        Create a head position mask
        
        Args:
            head_bbox: Head bounding box [x1, y1, x2, y2] (normalized 0-1)
            img_size: Size of the image
            
        Returns:
            head_pos: Head position mask tensor
        """
        width, height = img_size
        
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = head_bbox
        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            # Normalized coordinates, convert to pixel
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
        else:
            # Already in pixel coordinates, ensure integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Create mask
        head_mask = torch.zeros((height, width))
        
        # Ensure head bbox is within bounds
        x1 = max(0, min(width-1, x1))
        y1 = max(0, min(height-1, y1))
        x2 = max(x1+1, min(width, x2))
        y2 = max(y1+1, min(height, y2))
        
        # Set mask values
        head_mask[y1:y2, x1:x2] = 1.0
        
        # Add batch and channel dimensions
        head_mask = head_mask.unsqueeze(0)  # Add channel dimension
        
        return head_mask
    
    def crop_head(self, image, head_bbox, padding=0.2):
        """
        Crop the head region from the image
        
        Args:
            image: PIL Image
            head_bbox: Head bounding box [x1, y1, x2, y2] (normalized 0-1 or pixel coordinates)
            padding: Padding factor around the bbox
            
        Returns:
            head_img: Head crop tensor
        """
        width, height = image.size
        
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = head_bbox
        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            # Normalized coordinates, convert to pixel
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
        
        # Add padding
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        x1 = max(0, int(x1 - bbox_width * padding))
        y1 = max(0, int(y1 - bbox_height * padding))
        x2 = min(width, int(x2 + bbox_width * padding))
        y2 = min(height, int(y2 + bbox_height * padding))
        
        # Crop head region
        head_crop = image.crop((x1, y1, x2, y2))
        
        # Apply transform
        head_img = self.transform(head_crop)
        
        return head_img
    
    def create_objects_mask(self, objects=None, img_size=(224, 224)):
        """
        Create objects mask (placeholder - in a real application this would come from object detection)
        
        Args:
            objects: List of object bounding boxes and categories (optional)
            img_size: Size of the image
            
        Returns:
            objects_mask: Objects mask tensor with 11 channels for different categories
        """
        width, height = img_size
        num_categories = 11  # Using 11 categories as in the original model
        
        # Create empty objects mask
        objects_mask = torch.zeros((num_categories, height, width))
        
        # If objects are provided, fill the mask
        if objects:
            for obj in objects:
                category, bbox = obj['category'], obj['bbox']
                x1, y1, x2, y2 = bbox
                
                # Convert normalized coordinates if needed
                if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                    x1 = int(x1 * width)
                    y1 = int(y1 * height)
                    x2 = int(x2 * width)
                    y2 = int(y2 * height)
                
                # Ensure coordinates are within bounds
                x1 = max(0, min(width-1, x1))
                y1 = max(0, min(height-1, y1))
                x2 = max(x1+1, min(width, x2))
                y2 = max(y1+1, min(height, y2))
                
                # Set mask values
                objects_mask[category, y1:y2, x1:x2] = 1.0
        
        return objects_mask
    
    def predict(self, image_path, head_bbox, objects=None):
        """
        Run inference to predict gaze
        
        Args:
            image_path: Path to input image or PIL Image or numpy array
            head_bbox: Head bounding box [x1, y1, x2, y2] (normalized 0-1 or pixel coordinates)
            objects: List of object bounding boxes and categories (optional)
            
        Returns:
            gaze_heatmap: Predicted gaze heatmap
            in_frame_prob: Probability of gaze target being in frame
            visualization: Visualization of the prediction
        """
        # Preprocess image
        scene_img, original_image = self.preprocess_image(image_path)
        
        # Create head mask
        head_pos = self.create_head_position_mask(head_bbox)
        
        # Crop head region
        head_img = self.crop_head(original_image, head_bbox)
        
        # Create objects mask
        objects_mask = self.create_objects_mask(objects)
        
        # Move tensors to device
        scene_img = scene_img.unsqueeze(0).to(self.device)  # Add batch dimension
        head_img = head_img.unsqueeze(0).to(self.device)    # Add batch dimension
        head_pos = head_pos.unsqueeze(0).to(self.device)    # Add batch dimension
        objects_mask = objects_mask.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            heatmap, in_frame = self.model(scene_img, head_img, head_pos, objects_mask)
        
        # Convert outputs to numpy
        gaze_heatmap = heatmap.squeeze().cpu().numpy()
        in_frame_prob = torch.sigmoid(in_frame).item()
        
        # Create visualization
        visualization = self.visualize_prediction(
            original_image, head_bbox, gaze_heatmap, in_frame_prob
        )
        
        return gaze_heatmap, in_frame_prob, visualization
    
    def visualize_prediction(self, image, head_bbox, heatmap, in_frame_prob):
        """
        Visualize the gaze prediction
        
        Args:
            image: Original image (PIL Image)
            head_bbox: Head bounding box [x1, y1, x2, y2]
            heatmap: Predicted gaze heatmap
            in_frame_prob: Probability of gaze target being in frame
            
        Returns:
            visualization: Visualization image
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot original image with head bbox
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        
        # Draw bounding box
        width, height = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
        x1, y1, x2, y2 = head_bbox
        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            # Normalized coordinates, convert to pixel
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2))
        plt.title("Input Image")
        plt.axis('off')
        
        # Plot heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title(f"Gaze Heatmap (In-frame: {in_frame_prob:.2f})")
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(img_np)
        
        # Resize heatmap to match image size
        resized_heatmap = cv2.resize(heatmap, (width, height))
        plt.imshow(resized_heatmap, cmap='jet', alpha=0.5)
        plt.title("Gaze Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Convert figure to image
        fig = plt.gcf()
        plt.close()
        
        # Convert figure to image
        fig.canvas.draw()
        visualization = np.array(fig.canvas.renderer.buffer_rgba())
        
        return visualization
    
    def process_video(self, video_path, output_path=None, head_tracker=None, detector=None, sample_rate=1):
        """
        Process a video file and generate gaze predictions
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (None for no saving)
            head_tracker: Function that takes a frame and returns head bounding boxes
            detector: Function that takes a frame and returns object detections
            sample_rate: Process every Nth frame
            
        Returns:
            output_video_path: Path to output video
        """
        if head_tracker is None:
            raise ValueError("Head tracker function is required for video processing")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        processed_frames = 0
        
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_idx % sample_rate == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Get head bounding boxes
                    head_boxes = head_tracker(frame_rgb)
                    
                    # Get object detections
                    objects = detector(frame_rgb) if detector else None
                    
                    # Process each head
                    combined_heatmap = np.zeros((height, width))
                    
                    for head_bbox in head_boxes:
                        # Predict gaze
                        gaze_heatmap, in_frame_prob, _ = self.predict(frame_pil, head_bbox, objects)
                        
                        # Only add to combined heatmap if gaze is in frame
                        if in_frame_prob > 0.5:
                            # Resize heatmap to match frame size
                            resized_heatmap = cv2.resize(gaze_heatmap, (width, height))
                            combined_heatmap += resized_heatmap
                    
                    # Normalize combined heatmap
                    if np.max(combined_heatmap) > 0:
                        combined_heatmap = combined_heatmap / np.max(combined_heatmap)
                    
                    # Create colormap for visualization
                    heatmap_color = cv2.applyColorMap((combined_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    
                    # Blend with original frame
                    alpha = 0.5
                    blended = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
                    
                    # Draw head bounding boxes
                    for head_bbox in head_boxes:
                        x1, y1, x2, y2 = head_bbox
                        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                            # Normalized coordinates, convert to pixel
                            x1 = int(x1 * width)
                            y1 = int(y1 * height)
                            x2 = int(x2 * width)
                            y2 = int(y2 * height)
                        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Write frame to output video
                    if output_path:
                        out.write(blended)
                    
                    processed_frames += 1
                
                frame_idx += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
            print(f"Processed {processed_frames} frames. Output saved to {output_path}")
            return output_path
        
        return None


def main():
    parser = argparse.ArgumentParser(description='GESCAM Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to output directory')
    parser.add_argument('--head_bbox', type=float, nargs=4, default=[0.4, 0.4, 0.6, 0.6],
                       help='Head bounding box [x1, y1, x2, y2] (normalized 0-1)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.video:
        raise ValueError("Either --image or --video must be provided")
    
    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Initialize inference model
    inference = GazeInference(args.model)
    
    # Process image
    if args.image:
        gaze_heatmap, in_frame_prob, visualization = inference.predict(
            args.image, args.head_bbox
        )
        
        # Save visualization
        if args.output:
            output_path = os.path.join(args.output, 'gaze_prediction.png')
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGBA2BGRA))
            print(f"Visualization saved to {output_path}")
            
            # Save heatmap
            heatmap_path = os.path.join(args.output, 'gaze_heatmap.png')
            plt.imsave(heatmap_path, gaze_heatmap, cmap='jet')
            print(f"Heatmap saved to {heatmap_path}")
    
    # Process video
    if args.video:
        # Simple dummy head tracker - in a real application, you would use a face/head detector
        def dummy_head_tracker(frame):
            # Just return the head bbox provided in arguments
            return [args.head_bbox]
        
        # Simple dummy object detector - in a real application, you would use an object detector
        def dummy_object_detector(frame):
            return None
        
        # Output video path
        if args.output:
            video_name = os.path.basename(args.video)
            output_path = os.path.join(args.output, f"gaze_{video_name}")
        else:
            output_path = None
        
        # Process video
        inference.process_video(
            args.video,
            output_path=output_path,
            head_tracker=dummy_head_tracker,
            detector=dummy_object_detector,
            sample_rate=5  # Process every 5th frame for speed
        )

if __name__ == "__main__":
    main()