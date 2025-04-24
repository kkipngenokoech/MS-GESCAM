import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import cv2

def parse_xml(xml_path):
    """
    Parse XML annotation file to extract frame data.

    Args:
        xml_path (str): Path to the XML annotation file.

    Returns:
        dict: Dictionary mapping frame IDs to frame data (width, height, boxes, polylines).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = {}
    for frame in root.findall("frame"):
        frame_id = frame.find("filename").text
        width = int(frame.get("width", 1920))  # Default width if not specified
        height = int(frame.get("height", 1080))  # Default height if not specified
        boxes = [
            {
                "label": box.get("label"),
                "xtl": float(box.get("xtl", 0)),
                "ytl": float(box.get("ytl", 0)),
                "xbr": float(box.get("xbr", 0)),
                "ybr": float(box.get("ybr", 0))
            } for box in frame.findall("box")
        ]
        polylines = [
            {
                "label": polyline.get("label"),
                "points": polyline.get("points")
            } for polyline in frame.findall("polyline")
        ]
        frames[frame_id] = {
            "width": width,
            "height": height,
            "boxes": boxes,
            "polylines": polylines
        }
    return frames

def create_frame_to_image_mapping(image_folder):
    """
    Create a mapping from frame IDs to image paths.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        dict: Dictionary mapping frame IDs to image paths.
    """
    frame_to_image = {}
    for filename in os.listdir(image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            frame_id = filename
            frame_to_image[frame_id] = os.path.join(image_folder, filename)
    return frame_to_image

def extract_object_masks_from_annotations(frame_data, width, height, num_categories=11):
    """
    Extract object masks from frame annotations.

    Args:
        frame_data (dict): Frame data containing boxes and polylines.
        width (int): Frame width.
        height (int): Frame height.
        num_categories (int): Number of object categories.

    Returns:
        np.ndarray: Array of shape (num_categories, height, width) containing binary masks.
    """
    masks = np.zeros((num_categories, height, width), dtype=np.float32)
    for box in frame_data["boxes"]:
        label = box["label"].lower()
        category_idx = {
            "people": 0, "boards": 1, "books": 2, "monitors": 3, "phones": 4,
            "tables": 5, "water disp.": 6, "mugs": 7, "lamps": 8, "other1": 9, "other2": 10
        }.get(label, -1)
        if category_idx >= 0:
            x1, y1, x2, y2 = int(box["xtl"]), int(box["ytl"]), int(box["xbr"]), int(box["ybr"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            masks[category_idx, y1:y2, x1:x2] = 1.0
    return masks

def get_transforms(input_size=224, augment=True):
    """
    Get data transformations for training and validation.

    Args:
        input_size (int): Size to resize images to.
        augment (bool): Whether to apply data augmentation.

    Returns:
        transforms.Compose: Transformation pipeline.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if augment:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])
    return transform

class GESCAMCustomDataset(Dataset):
    """
    Dataset class for GESCAM (Gaze Estimation based Synthetic Classroom Attention Measurement).
    Customized for the specific annotation format.
    """
    def __init__(self, xml_path, image_folder, transform=None, head_transform=None,
                 input_size=224, output_size=64, test=False):
        """
        Args:
            xml_path (str): Path to the XML annotation file.
            image_folder (str): Path to the folder containing images.
            transform: Transformations to apply to the scene image.
            head_transform: Transformations to apply to the head crop.
            input_size: Input image size for the model.
            output_size: Output heatmap size.
            test: Whether this is a test dataset.
        """
        super(GESCAMCustomDataset, self).__init__()

        self.xml_path = xml_path
        self.image_folder = image_folder
        self.transform = transform
        self.head_transform = head_transform if head_transform else transform
        self.input_size = input_size
        self.output_size = output_size
        self.test = test

        # Parse annotations and create image mapping
        self.frames = parse_xml(xml_path)
        self.frame_to_image = create_frame_to_image_mapping(image_folder)

        # Create samples
        self.samples = self._create_samples()
        print(f"Created dataset with {len(self.samples)} samples")

    def _match_person_to_sight_line(self, person_box, polylines):
        """
        Match a person bounding box to the corresponding line of sight polyline.

        Args:
            person_box: Dictionary containing person bounding box.
            polylines: List of polyline dictionaries for the frame.

        Returns:
            target_point: (x,y) tuple of gaze target or None if no match.
            has_target: Boolean indicating if a match was found.
        """
        sight_lines = [p for p in polylines if p["label"].lower() == "line of sight"]
        if not sight_lines:
            return None, False

        person_center_x = (person_box["xtl"] + person_box["xbr"]) / 2
        person_center_y = (person_box["ytl"] + person_box["ybr"]) / 2
        person_width = person_box["xbr"] - person_box["xtl"]

        best_match = None
        best_distance = float('inf')

        for polyline in sight_lines:
            points_str = polyline["points"]
            try:
                points = [tuple(map(float, point.split(","))) for point in points_str.split(";")]
                if len(points) >= 2:
                    start_x, start_y = points[0]
                    end_x, end_y = points[-1]
                    distance = np.sqrt((start_x - person_center_x)**2 + (start_y - person_center_y)**2)
                    if distance < best_distance and distance < person_width * 1.5:
                        best_distance = distance
                        best_match = (end_x, end_y)
            except Exception as e:
                print(f"Error parsing polyline points: {e}, points_str: {points_str}")
                continue

        return best_match, best_match is not None

    def _create_samples(self):
        """
        Create dataset samples from parsed frames.

        Returns:
            samples: List of sample dictionaries.
        """
        samples = []
        frames_with_persons = 0
        frames_with_sight_lines = 0

        for frame_id, frame_data in self.frames.items():
            if frame_id not in self.frame_to_image:
                continue

            image_path = self.frame_to_image[frame_id]
            width, height = frame_data["width"], frame_data["height"]
            object_masks = extract_object_masks_from_annotations(frame_data, width, height)

            person_boxes = [box for box in frame_data["boxes"] if "person" in box["label"].lower()]
            if person_boxes:
                frames_with_persons += 1

            sight_lines = [p for p in frame_data["polylines"] if p["label"].lower() == "line of sight"]
            if sight_lines:
                frames_with_sight_lines += 1

            for person_box in person_boxes:
                gaze_target, has_target = self._match_person_to_sight_line(person_box, frame_data["polylines"])
                sample = {
                    "frame_id": frame_id,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "head_bbox": [person_box["xtl"], person_box["ytl"], person_box["xbr"], person_box["ybr"]],
                    "gaze_target": gaze_target,
                    "in_frame": has_target,
                    "object_masks": object_masks
                }
                samples.append(sample)

        print(f"Statistics: {frames_with_persons} frames with person boxes, {frames_with_sight_lines} frames with sight lines")
        return samples

    def _create_head_position_channel(self, head_bbox, width, height):
        """
        Create a binary mask for head position.
        """
        x1, y1, x2, y2 = head_bbox
        head_mask = torch.zeros(height, width)
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(width, x2)), int(min(height, y2))
        head_mask[y1:y2, x1:x2] = 1.0
        return head_mask

    def _create_gaze_heatmap(self, gaze_target, width, height):
        """
        Create a Gaussian heatmap at the gaze point.
        """
        if not gaze_target:
            return torch.zeros(self.output_size, self.output_size)

        x, y = gaze_target
        x = x * self.output_size / width
        y = y * self.output_size / height
        Y, X = torch.meshgrid(torch.arange(self.output_size), torch.arange(self.output_size), indexing='ij')
        sigma = 3.0
        heatmap = torch.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
        return heatmap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            img = Image.open(sample["image_path"]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            img = Image.new('RGB', (self.input_size, self.input_size), color='gray')

        width, height = sample["width"], sample["height"]
        head_bbox = sample["head_bbox"]
        x1, y1, x2, y2 = head_bbox
        x1 = max(0, min(width-1, x1))
        y1 = max(0, min(height-1, y1))
        x2 = max(x1+1, min(width, x2))
        y2 = max(y1+1, min(height, y2))

        try:
            head_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        except Exception as e:
            print(f"Error cropping head: {e}, bbox: {head_bbox}, image size: {img.size}")
            head_img = Image.new('RGB', (100, 100), color='gray')

        head_pos = self._create_head_position_channel(head_bbox, width, height)
        object_masks = torch.from_numpy(sample["object_masks"])

        if sample["in_frame"] and sample["gaze_target"]:
            gaze_target = sample["gaze_target"]
            gaze_heatmap = self._create_gaze_heatmap(gaze_target, width, height)
            head_center_x = (x1 + x2) / 2 / width
            head_center_y = (y1 + y2) / 2 / height
            gaze_x = gaze_target[0] / width
            gaze_y = gaze_target[1] / height
            gaze_vector = torch.tensor([gaze_x - head_center_x, gaze_y - head_center_y])
        else:
            gaze_heatmap = torch.zeros(self.output_size, self.output_size)
            gaze_vector = torch.tensor([0.0, 0.0])

        if self.transform:
            img = self.transform(img)
        if self.head_transform:
            head_img = self.head_transform(head_img)

        head_pos = head_pos.unsqueeze(0)
        head_pos = F.interpolate(head_pos.unsqueeze(0), size=(self.input_size, self.input_size),
                                 mode='nearest').squeeze(0)
        object_masks = F.interpolate(object_masks.unsqueeze(0),
                                     size=(self.input_size, self.input_size),
                                     mode='nearest').squeeze(0)

        in_frame = torch.tensor([float(sample["in_frame"])])
        object_label = torch.tensor([0])

        metadata = {
            "frame_id": sample["frame_id"],
            "image_path": sample["image_path"],
            "head_bbox": sample["head_bbox"],
            "original_size": (width, height)
        }

        return img, head_img, head_pos, gaze_heatmap, in_frame, object_label, gaze_vector, object_masks, metadata

def visualize_sample_with_objects(sample, save_path=None):
    """
    Visualize a dataset sample with object masks.

    Args:
        sample: Tuple of tensors from dataset __getitem__.
        save_path: Path to save visualization (if None, displays inline).
    """
    img, head_img, head_pos, gaze_heatmap, in_frame, object_label, gaze_vector, object_masks, metadata = sample

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_vis = img.clone()
    img_vis = img_vis * std + mean
    img_vis = img_vis.permute(1, 2, 0).numpy()
    img_vis = np.clip(img_vis, 0, 1)

    head_img_vis = head_img.clone()
    head_img_vis = head_img_vis * std + mean
    head_img_vis = head_img_vis.permute(1, 2, 0).numpy()
    head_img_vis = np.clip(head_img_vis, 0, 1)

    plt.figure(figsize=(15, 12))

    plt.subplot(3, 3, 1)
    plt.imshow(img_vis)
    plt.title(f"Frame ID: {metadata['frame_id']}")
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(head_img_vis)
    plt.title("Head/Person Crop")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(head_pos.squeeze().numpy(), cmap='gray')
    plt.title("Head Position Channel")
    plt.axis('off')

    category_names = ["People", "Boards", "Books", "Monitors", "Phones",
                     "Tables", "Water Disp.", "Mugs", "Lamps", "Other1", "Other2"]
    for i in range(min(4, object_masks.shape[0])):
        plt.subplot(3, 3, 4 + i)
        plt.imshow(object_masks[i].numpy(), cmap='viridis')
        plt.title(f"Object: {category_names[i]}")
        plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(gaze_heatmap.numpy(), cmap='jet')
    plt.title(f"Gaze Heatmap (In-frame: {bool(in_frame.item())})")
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(img_vis)
    heatmap_vis = gaze_heatmap.numpy()
    heatmap_vis = cv2.resize(heatmap_vis, (img_vis.shape[1], img_vis.shape[0]))
    if in_frame.item():
        plt.imshow(heatmap_vis, cmap='jet', alpha=0.5)
    plt.title("Heatmap Overlay")
    plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_dataset(xml_path, image_folder):
    """
    Test the dataset with visualization.

    Args:
        xml_path: Path to the XML annotation file.
        image_folder: Path to the folder with images.
    """
    transform = get_transforms(augment=False)
    dataset = GESCAMCustomDataset(
        xml_path=xml_path,
        image_folder=image_folder,
        transform=transform,
        head_transform=transform,
        input_size=224,
        output_size=64,
        test=True
    )
    
    # Visualize a few samples
    num_samples_to_visualize = min(5, len(dataset))
    for i in range(num_samples_to_visualize):
        sample = dataset[i]
        save_path = f"sample_visualization_{i}.png"
        visualize_sample_with_objects(sample, save_path=save_path)
        print(f"Saved visualization for sample {i} to {save_path}")