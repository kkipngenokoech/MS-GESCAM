import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

# Define the model architecture (same as your training script)
class SoftAttention(torch.nn.Module):
    """
    Soft attention module for attending to scene features based on head features
    """
    def __init__(self, head_channels=256, output_size=(7, 7)):
        super(SoftAttention, self).__init__()
        self.output_h, self.output_w = output_size

        # Attention layers
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(head_channels, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.output_h * self.output_w),
            torch.nn.Sigmoid()
        )

    def forward(self, head_features):
        # Input head_features shape: [batch_size, head_channels]
        batch_size = head_features.size(0)

        # Generate attention weights
        attn_weights = self.attention(head_features)

        # Reshape to spatial attention map
        attn_weights = attn_weights.view(batch_size, 1, self.output_h, self.output_w)

        return attn_weights


class MSGESCAMModel(torch.nn.Module):
    """
    Multi-Stream GESCAM architecture for gaze estimation in classroom settings
    """
    def __init__(self, pretrained=False, output_size=64):
        super(MSGESCAMModel, self).__init__()

        # Store the output size
        self.output_size = output_size

        # Feature dimensions
        self.backbone_dim = 512  # ResNet18 outputs 512 feature channels
        self.feature_dim = 256

        # Downsampled feature map size
        self.map_size = 7  # ResNet outputs 7x7 feature maps

        # === Scene Pathway ===
        # Load a pre-trained ResNet18 without the final layer
        self.scene_backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)

        # Save the original conv1 weights
        original_conv1_weight = self.scene_backbone.conv1.weight.clone()

        # Create a new conv1 layer that accepts 4 channels (RGB + head position)
        self.scene_backbone.conv1 = torch.nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize with the pre-trained weights
        with torch.no_grad():
            self.scene_backbone.conv1.weight[:, :3] = original_conv1_weight
            # Initialize the new channel with small random values
            self.scene_backbone.conv1.weight[:, 3] = 0.01 * torch.randn_like(self.scene_backbone.conv1.weight[:, 0])

        # Remove the final FC layer from the scene backbone
        self.scene_features = torch.nn.Sequential(*list(self.scene_backbone.children())[:-1])

        # Add a FC layer to transform from backbone_dim to feature_dim
        self.scene_fc = torch.nn.Linear(self.backbone_dim, self.feature_dim)

        # === Head Pathway ===
        # Load another pre-trained ResNet18 for the head pathway
        self.head_backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)

        # Remove the final FC layer from the head backbone
        self.head_features = torch.nn.Sequential(*list(self.head_backbone.children())[:-1])

        # Add a FC layer to transform from backbone_dim to feature_dim
        self.head_fc = torch.nn.Linear(self.backbone_dim, self.feature_dim)

        # === Objects Mask Enhancement (optional) ===
        # This takes an objects mask (with channels for different object classes)
        self.objects_conv = torch.nn.Conv2d(11, 512, kernel_size=3, stride=2, padding=1)  # 11 object categories

        # Soft attention mechanism
        self.attention = SoftAttention(head_channels=self.feature_dim, output_size=(self.map_size, self.map_size))

        # === Fusion and Encoding ===
        # Fusion of attended scene features and head features
        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(self.backbone_dim + self.feature_dim, self.feature_dim, kernel_size=1),
            torch.nn.BatchNorm2d(self.feature_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.feature_dim),
            torch.nn.ReLU(inplace=True)
        )

        # Calculate the number of deconvolution layers needed
        # Each layer doubles the size, so we need log2(output_size / 7) layers
        import math
        self.num_deconv_layers = max(1, int(math.log2(output_size / 7)) + 1)

        # === Decoding for heatmap generation ===
        deconv_layers = []
        in_channels = self.feature_dim
        out_size = self.map_size

        # Create deconvolution layers
        for i in range(self.num_deconv_layers - 1):
            # Calculate output channels
            out_channels = max(32, in_channels // 2)

            # Add deconv layer
            deconv_layers.extend([
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            ])

            in_channels = out_channels
            out_size *= 2

        # Final layer to adjust to exact output size
        if out_size != output_size:
            # Add a final layer with correct stride to reach exactly output_size
            scale_factor = output_size / out_size
            stride = 2 if scale_factor > 1 else 1
            output_padding = 1 if scale_factor > 1 else 0

            deconv_layers.extend([
                torch.nn.ConvTranspose2d(
                    in_channels, 1, kernel_size=3,
                    stride=stride, padding=1, output_padding=output_padding
                )
            ])
        else:
            # If we're already at the right size, just add a 1x1 conv
            deconv_layers.append(torch.nn.Conv2d(in_channels, 1, kernel_size=1))

        self.decode = torch.nn.Sequential(*deconv_layers)

        # === In-frame probability prediction ===
        self.in_frame_fc = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 1)
        )

    def forward(self, scene_img, head_img, head_pos, objects_mask=None):
        """
        Forward pass through the MS-GESCAM network

        Args:
            scene_img: Scene image tensor [batch_size, 3, H, W]
            head_img: Head crop tensor [batch_size, 3, H, W]
            head_pos: Head position mask [batch_size, 1, H, W]
            objects_mask: Optional mask of object categories [batch_size, num_categories, H, W]

        Returns:
            heatmap: Predicted gaze heatmap [batch_size, 1, output_size, output_size]
            in_frame: Probability of gaze target being in frame [batch_size, 1]
        """
        batch_size = scene_img.size(0)

        # === Process scene pathway ===
        # Concatenate scene image and head position channel
        scene_input = torch.cat([scene_img, head_pos], dim=1)

        # Process through ResNet layers until layer4 (skipping the final global pooling and FC)
        x = self.scene_backbone.conv1(scene_input)
        x = self.scene_backbone.bn1(x)
        x = self.scene_backbone.relu(x)
        x = self.scene_backbone.maxpool(x)

        x = self.scene_backbone.layer1(x)
        x = self.scene_backbone.layer2(x)
        x = self.scene_backbone.layer3(x)
        scene_features_map = self.scene_backbone.layer4(x)  # [batch_size, 512, 7, 7]

        # Global average pooling for scene features
        scene_vector = torch.nn.functional.adaptive_avg_pool2d(scene_features_map, (1, 1)).view(batch_size, -1)
        scene_features = self.scene_fc(scene_vector)  # [batch_size, feature_dim]

        # === Process head pathway ===
        # Process through the entire head features extractor
        head_vector = self.head_features(head_img).view(batch_size, -1)  # [batch_size, 512]
        head_features = self.head_fc(head_vector)  # [batch_size, feature_dim]

        # Process objects mask if provided
        if objects_mask is not None:
            obj_features = self.objects_conv(objects_mask)
            # Resize to match scene features map if needed
            if obj_features.size(2) != scene_features_map.size(2):
                obj_features = torch.nn.functional.adaptive_avg_pool2d(
                    obj_features, (scene_features_map.size(2), scene_features_map.size(3))
                )
            # Add object features to scene features
            scene_features_map = scene_features_map + obj_features

        # Generate attention map from head features
        attn_weights = self.attention(head_features)  # [batch_size, 1, 7, 7]

        # Apply attention to scene features map
        attended_scene = scene_features_map * attn_weights  # [batch_size, 512, 7, 7]

        # Reshape head features to concatenate with scene features
        head_features_map = head_features.view(batch_size, self.feature_dim, 1, 1)
        head_features_map = head_features_map.expand(-1, -1, self.map_size, self.map_size)

        # Concatenate attended scene features and head features
        concat_features = torch.cat([attended_scene, head_features_map], dim=1)  # [batch_size, 512+256, 7, 7]

        # Encode the concatenated features
        encoded = self.encode(concat_features)  # [batch_size, 256, 7, 7]

        # Predict in-frame probability
        in_frame = self.in_frame_fc(head_features)

        # Decode to get the final heatmap
        heatmap = self.decode(encoded)

        # Ensure output size is correct
        if heatmap.size(2) != self.output_size or heatmap.size(3) != self.output_size:
            heatmap = torch.nn.functional.interpolate(
                heatmap,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=True
            )

        # Apply sigmoid to get values between 0 and 1
        heatmap = torch.sigmoid(heatmap)

        return heatmap, in_frame


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
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                         fill=False, edgecolor='green', linewidth=2))
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