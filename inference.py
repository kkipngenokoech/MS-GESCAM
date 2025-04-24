import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from modelArch import ModelLoader

# Assuming GESCAMCustomDataset and get_transforms are defined elsewhere
# from your_dataset import GESCAMCustomDataset, get_transforms


class Preprocessor:
    """Handles preprocessing of images and masks for model input."""
    
    def __init__(self, transform=None, scene_size=(224, 224), num_categories=11):
        self.scene_size = scene_size
        self.num_categories = num_categories
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(self.scene_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess a scene image."""
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
    
    def preprocess_head_image(self, head_image_path):
        """Load and preprocess a head image."""
        head_image = Image.open(head_image_path).convert("RGB")
        return self.transform(head_image)
    
    def create_head_position_mask(self, bbox):
        """Create a head position mask from bounding box coordinates."""
        mask = np.zeros(self.scene_size, dtype=np.float32)
        x_min, y_min, x_max, y_max = bbox
        mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1.0
        mask = cv2.resize(mask, self.scene_size)
        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    
    def preprocess_objects_mask(self, objects_mask_path):
        """Load and preprocess an objects mask (if available)."""
        if objects_mask_path and os.path.exists(objects_mask_path):
            objects_mask = np.load(objects_mask_path)  # Assuming .npy format
            objects_mask = cv2.resize(objects_mask, self.scene_size)
            return torch.tensor(objects_mask, dtype=torch.float32).permute(2, 0, 1)
        return None

    def denormalize_image(self, img_tensor):
        """Denormalize an image tensor for visualization."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_vis = img_tensor.clone()
        img_vis = img_vis * std + mean
        img_vis = img_vis.permute(1, 2, 0).numpy()
        return np.clip(img_vis, 0, 1)

class Evaluator:
    """Handles evaluation metrics calculation."""
    
    @staticmethod
    def calculate_auc(pred_heatmap, target_heatmap):
        """Calculate Area Under the ROC Curve for heatmap prediction."""
        pred_flat = pred_heatmap.flatten()
        target_flat = (target_heatmap > 0.1).flatten().astype(int)
        fpr, tpr, _ = roc_curve(target_flat, pred_flat)
        return auc(fpr, tpr)
    
    @staticmethod
    def calculate_distance_error(pred_heatmap, target_heatmap, normalize=True):
        """Calculate distance error between predicted and target gaze points."""
        pred_idx = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        target_idx = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
        y1, x1 = pred_idx
        y2, x2 = target_idx
        if normalize:
            h, w = target_heatmap.shape
            x1, y1 = x1 / w, y1 / h
            x2, y2 = x2 / w, y2 / h
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    @staticmethod
    def calculate_angular_error(pred_vector, target_vector):
        """Calculate angular error between predicted and target gaze vectors."""
        pred_norm = np.linalg.norm(pred_vector)
        target_norm = np.linalg.norm(target_vector)
        if pred_norm < 1e-7 or target_norm < 1e-7:
            return 180.0
        pred_normalized = pred_vector / pred_norm
        target_normalized = target_vector / target_norm
        dot_product = np.clip(np.dot(pred_normalized, target_normalized), -1.0, 1.0)
        return np.arccos(dot_product) * 180 / np.pi
    
    @staticmethod
    def calculate_in_frame_accuracy(pred_in_frame, target_in_frame, threshold=0.5):
        """Calculate accuracy of in-frame prediction."""
        pred_binary = (pred_in_frame > threshold).astype(int)
        target_binary = target_in_frame.astype(int)
        return (pred_binary == target_binary).mean()
    
    @staticmethod
    def extract_gaze_vector_from_heatmap(heatmap, head_center, heatmap_size, normalize=True):
        """Extract gaze vector from heatmap peak."""
        peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        peak_y, peak_x = peak_idx
        h, w = heatmap.shape
        peak_x_norm = peak_x / w
        peak_y_norm = peak_y / h
        gaze_vector = np.array([peak_x_norm - head_center[0], peak_y_norm - head_center[1]])
        if normalize and np.linalg.norm(gaze_vector) > 0:
            gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        return gaze_vector
    
    @staticmethod
    def visualize_prediction(img, head_bbox, pred_heatmap, target_heatmap, pred_in_frame, target_in_frame, save_path=None):
        """Visualize model prediction versus ground truth."""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        x1, y1, x2, y2 = head_bbox
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=2))
        plt.title(f"Head (Target In-frame: {bool(target_in_frame)})")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(pred_heatmap, cmap='jet')
        plt.title(f"Predicted Heatmap (P={pred_in_frame:.2f})")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(target_heatmap, cmap='jet')
        plt.title("Ground Truth Heatmap")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(img)
        plt.imshow(pred_heatmap, cmap='jet', alpha=0.5)
        plt.title("Predicted Overlay")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(img)
        plt.imshow(target_heatmap, cmap='jet', alpha=0.5)
        plt.title("Ground Truth Overlay")
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        error_map = np.abs(pred_heatmap - target_heatmap)
        plt.imshow(error_map, cmap='hot')
        plt.title("Prediction Error")
        plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class Validator:
    """Handles model validation and visualization."""
    
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.evaluator = Evaluator()
    
    def validate(self, dataset, batch_size=8, num_vis=20, vis_dir=None):
        """Validate model performance on dataset."""
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
        
        self.model.eval()
        all_auc = []
        all_distance = []
        all_angular = []
        all_in_frame_acc = []
        all_pred_in_frame = []
        all_target_in_frame = []
        
        vis_indices = np.random.choice(len(dataset), min(num_vis, len(dataset)), replace=False) if len(dataset) > 0 and num_vis > 0 else []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Validating")):
                scene_img, head_img, head_pos, target_heatmap, target_in_frame, _, target_vector, object_masks, metadata = batch
                scene_img = scene_img.to(self.device)
                head_img = head_img.to(self.device)
                head_pos = head_pos.to(self.device)
                object_masks = object_masks.to(self.device)
                
                pred_heatmap, pred_in_frame = self.model(scene_img, head_img, head_pos, object_masks)
                pred_heatmap = pred_heatmap.squeeze(1).cpu().numpy()
                pred_in_frame_prob = torch.sigmoid(pred_in_frame).squeeze().cpu().numpy()
                target_heatmap_np = target_heatmap.cpu().numpy()
                target_in_frame_np = target_in_frame.squeeze().cpu().numpy()
                
                for i in range(len(scene_img)):
                    if target_in_frame_np[i] > 0.5:
                        auc_score = self.evaluator.calculate_auc(pred_heatmap[i], target_heatmap_np[i])
                        all_auc.append(auc_score)
                        dist_error = self.evaluator.calculate_distance_error(pred_heatmap[i], target_heatmap_np[i])
                        all_distance.append(dist_error)
                        pred_vector = target_vector[i].cpu().numpy()  # Placeholder; replace with heatmap-based vector if needed
                        target_vec = target_vector[i].cpu().numpy()
                        angular_error = self.evaluator.calculate_angular_error(pred_vector, target_vec)
                        all_angular.append(angular_error)
                    
                    all_pred_in_frame.append(pred_in_frame_prob[i])
                    all_target_in_frame.append(target_in_frame_np[i])
        
        if vis_dir:
            for vis_idx, idx in enumerate(tqdm(vis_indices, desc="Generating visualizations")):
                sample = dataset[idx]
                scene_img, head_img, head_pos, target_heatmap, target_in_frame, _, _, object_masks, metadata = sample
                scene_img_batch = scene_img.unsqueeze(0).to(self.device)
                head_img_batch = head_img.unsqueeze(0).to(self.device)
                head_pos_batch = head_pos.unsqueeze(0).to(self.device)
                object_masks_batch = object_masks.unsqueeze(0).to(self.device)
                
                pred_heatmap, pred_in_frame = self.model(scene_img_batch, head_img_batch, head_pos_batch, object_masks_batch)
                pred_heatmap_np = pred_heatmap.squeeze().cpu().numpy()
                pred_in_frame_prob = torch.sigmoid(pred_in_frame).item()
                
                img_vis = self.preprocessor.denormalize_image(scene_img)
                x1, y1, x2, y2 = metadata['head_bbox']
                h, w = img_vis.shape[:2]
                orig_w, orig_h = metadata['original_size']
                scale_x, scale_y = w / orig_w, h / orig_h
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                
                vis_path = os.path.join(vis_dir, f"validation_{vis_idx}.png")
                self.evaluator.visualize_prediction(
                    img_vis, [x1, y1, x2, y2], pred_heatmap_np, target_heatmap.numpy(),
                    pred_in_frame_prob, target_in_frame.item(), vis_path
                )
        
        in_frame_accuracy = self.evaluator.calculate_in_frame_accuracy(
            np.array(all_pred_in_frame), np.array(all_target_in_frame)
        )
        
        metrics = {
            'auc_mean': np.mean(all_auc) if all_auc else np.nan,
            'auc_std': np.std(all_auc) if all_auc else np.nan,
            'distance_mean': np.mean(all_distance) if all_distance else np.nan,
            'distance_std': np.std(all_distance) if all_distance else np.nan,
            'angular_mean': np.mean(all_angular) if all_angular else np.nan,
            'angular_std': np.std(all_angular) if all_angular else np.nan,
            'in_frame_accuracy': in_frame_accuracy,
            'num_evaluated': len(all_auc)
        }
        
        return metrics

class HeatmapVideoCreator:
    """Creates an attention heatmap video for multiple frames."""
    
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
    
    def create_attention_heatmap(self, dataset, output_path, frame_indices=None, num_frames=20):
        """Create an attention heatmap visualization for entire frames."""
        all_frame_ids = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            metadata = sample[8]
            frame_id = metadata['frame_id']
            if frame_id not in all_frame_ids:
                all_frame_ids.append(frame_id)
        
        if frame_indices is None:
            frame_indices = sorted(np.random.choice(all_frame_ids, min(num_frames, len(all_frame_ids)), replace=False))
        
        temp_dir = "temp_attention_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, frame_id in enumerate(tqdm(frame_indices, desc="Creating attention heatmaps")):
            frame_samples = [idx for idx in range(len(dataset)) if dataset[idx][8]['frame_id'] == frame_id]
            if not frame_samples:
                continue
            
            first_sample = dataset[frame_samples[0]]
            scene_img, _, _, _, _, _, _, object_masks, metadata = first_sample
            img_size = scene_img.shape[1:3]
            img_vis = self.preprocessor.denormalize_image(scene_img)
            combined_heatmap = np.zeros(img_size)
            
            with torch.no_grad():
                for sample_idx in frame_samples:
                    sample = dataset[sample_idx]
                    scene_img, head_img, head_pos, _, target_in_frame, _, _, object_masks, _ = sample
                    if target_in_frame.item() < 0.5:
                        continue
                    
                    scene_img_batch = scene_img.unsqueeze(0).to(self.device)
                    head_img_batch = head_img.unsqueeze(0).to(self.device)
                    head_pos_batch = head_pos.unsqueeze(0).to(self.device)
                    object_masks_batch = object_masks.unsqueeze(0).to(self.device)
                    
                    pred_heatmap, _ = self.model(scene_img_batch, head_img_batch, head_pos_batch, object_masks_batch)
                    pred_heatmap_np = pred_heatmap.squeeze().cpu().numpy()
                    pred_heatmap_resized = cv2.resize(pred_heatmap_np, (img_size[1], img_size[0]))
                    combined_heatmap += pred_heatmap_resized
            
            if np.max(combined_heatmap) > 0:
                combined_heatmap = combined_heatmap / np.max(combined_heatmap)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img_vis)
            plt.imshow(combined_heatmap, cmap='jet', alpha=0.5)
            plt.title(f"Frame {frame_id} - Combined Attention")
            plt.axis('off')
            
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_path)
            plt.close()
        
        frame_paths = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.png')])
        if not frame_paths:
            print("No frames generated!")
            return
        
        first_frame = cv2.imread(frame_paths[0])
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 2, (width, height))
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
        
        video_writer.release()
        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(temp_dir)
        
        print(f"Attention heatmap video saved to {output_path}")

class AttentionAnalyzer:
    """Main class to orchestrate the attention analysis pipeline."""
    
    def __init__(self, model_path, output_dir="validation_results"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model_loader = ModelLoader(model_path, output_size=64, device=self.device)
        self.preprocessor = Preprocessor(transform=get_transforms(augment=False))
        self.validator = Validator(self.model_loader.get_model(), self.preprocessor, self.device)
        self.heatmap_creator = HeatmapVideoCreator(self.model_loader.get_model(), self.preprocessor, self.device)
    
    def run_validation(self, dataset, batch_size=8, num_vis=20):
        """Run model validation and generate visualizations."""
        val_size = min(int(0.2 * len(dataset)), 500)
        generator = torch.Generator().manual_seed(42)
        _, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=generator)
        print(f"Validation dataset size: {len(val_dataset)}")
        
        vis_dir = os.path.join(self.output_dir, "visualizations")
        metrics = self.validator.validate(dataset=val_dataset, batch_size=batch_size, num_vis=num_vis, vis_dir=vis_dir)
        
        print("\nModel Validation Metrics:")
        print("-" * 30)
        print(f"AUC: {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
        print(f"Distance Error: {metrics['distance_mean']:.4f} ± {metrics['distance_std']:.4f}")
        print(f"Angular Error: {metrics['angular_mean']:.2f}° ± {metrics['angular_std']:.2f}°")
        print(f"In-frame Accuracy: {metrics['in_frame_accuracy']:.4f}")
        print(f"Number of evaluated samples: {metrics['num_evaluated']}")
        
        metrics_path = os.path.join(self.output_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        return metrics
    
    def create_heatmap_video(self, dataset, num_frames=20):
        """Create an attention heatmap video."""
        heatmap_video_path = os.path.join(self.output_dir, "attention_heatmap.mp4")
        self.heatmap_creator.create_attention_heatmap(
            dataset=dataset, output_path=heatmap_video_path, num_frames=num_frames
        )

def main():
    base_dir = "path/to/your/data"  # Update with your data directory
    output_dir = "gescam_output"
    model_path = f"{output_dir}/best_model.pt"
    xml_path = f"{base_dir}/test_subset/task_classroom_11_video-01_final/annotations.xml"
    image_folder = f"{base_dir}/test_subset/task_classroom_11_video-01_final/images"
    
    dataset = GESCAMCustomDataset(xml_path=xml_path, image_folder=image_folder, transform=get_transforms(augment=False))
    analyzer = AttentionAnalyzer(model_path, output_dir)
    
    print("Validating model...")
    analyzer.run_validation(dataset, batch_size=8, num_vis=20)
    
    print("Creating attention heatmap video...")
    analyzer.create_heatmap_video(dataset, num_frames=20)
    
    print(f"Validation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()