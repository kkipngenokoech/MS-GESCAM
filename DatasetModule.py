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
import re
from tqdm import tqdm

def extract_object_masks_from_annotations(frame_data, width, height, num_categories=11):
    """
    Extract object masks from frame annotations

    Args:
        frame_data: Dictionary containing frame annotations
        width: Image width
        height: Image height
        num_categories: Number of object categories

    Returns:
        object_masks: Array of shape [num_categories, height, width]
    """
    # Initialize masks
    object_masks = np.zeros((num_categories, height, width), dtype=np.float32)

    # Map of object labels to category indices
    category_map = {
        'person': 0,
        'teacher': 0,  # Group all people in category 0
        'blackboard': 1,
        'whiteboard': 1,  # Group all boards in category 1
        'notebook': 2,
        'book': 2,  # Group all reading materials in category 2
        'monitor': 3,
        'screen': 3,  # Group all displays in category 3
        'mobile': 4,
        'phone': 4,  # Group all phones in category 4
        'table': 5,
        'desk': 5,  # Group all tables in category 5
        'water dispenser': 6,
        'mug': 7,
        'table lamp': 8,
        # Add more mappings as needed
    }

    # Process each box
    for box in frame_data["boxes"]:
        label = box["label"].lower()

        # Extract category
        category = -1
        for key, idx in category_map.items():
            if key in label:
                category = idx
                break

        if category >= 0 and category < num_categories:
            # Extract coordinates
            x1, y1 = int(box["xtl"]), int(box["ytl"])
            x2, y2 = int(box["xbr"]), int(box["ybr"])

            # Ensure within bounds
            x1 = max(0, min(width-1, x1))
            y1 = max(0, min(height-1, y1))
            x2 = max(x1+1, min(width, x2))
            y2 = max(y1+1, min(height, y2))

            # Set mask
            object_masks[category, y1:y2, x1:x2] = 1.0

    return object_masks

def parse_xml(xml_path):
    """
    Parse the XML file and extract bounding boxes and polylines for each frame.

    Args:
        xml_path (str): Path to the XML file.

    Returns:
        frames (dict): Dictionary containing frame data (bounding boxes and polylines).
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        frames = {}

        print(f"Parsing XML annotations from {xml_path}")
        print(f"Root tag: {root.tag}, with {len(root)} child elements")

        frame_count = 0
        for image in tqdm(root.findall("image"), desc="Parsing frames"):
            try:
                frame_id = int(image.attrib["id"])  # Extract frame ID
                frame_name = image.attrib["name"]  # Extract frame name
                width = int(image.attrib["width"])  # Image width
                height = int(image.attrib["height"])  # Image height

                # Initialize lists for bounding boxes and polylines
                frame_boxes = []
                frame_polylines = []

                # Extract bounding boxes
                for box in image.findall("box"):
                    try:
                        # Extract all box attributes
                        box_info = {
                            "label": box.attrib.get("label", "unknown"),
                            "xtl": float(box.attrib.get("xtl", 0)),
                            "ytl": float(box.attrib.get("ytl", 0)),
                            "xbr": float(box.attrib.get("xbr", 0)),
                            "ybr": float(box.attrib.get("ybr", 0)),
                        }
                        frame_boxes.append(box_info)
                    except Exception as box_err:
                        print(f"Error parsing box in frame {frame_id}: {box_err}")

                # Extract polylines
                for polyline in image.findall("polyline"):
                    try:
                        polyline_info = {
                            "label": polyline.attrib.get("label", "unknown"),
                            "points": polyline.attrib.get("points", "")
                        }
                        frame_polylines.append(polyline_info)
                    except Exception as polyline_err:
                        print(f"Error parsing polyline in frame {frame_id}: {polyline_err}")

                # Store frame information
                frames[frame_id] = {
                    "name": frame_name,
                    "width": width,
                    "height": height,
                    "boxes": frame_boxes,
                    "polylines": frame_polylines
                }
                frame_count += 1

                # Debug first frame
                if frame_count == 1:
                    print(f"Sample frame: ID={frame_id}, Name={frame_name}, Size={width}x{height}")
                    print(f"Found {len(frame_boxes)} boxes and {len(frame_polylines)} polylines in first frame")
                    if frame_boxes:
                        print(f"Sample box labels: {[box['label'] for box in frame_boxes[:5]]}")
                    if frame_polylines:
                        print(f"Sample polyline labels: {[p['label'] for p in frame_polylines[:5]]}")
            except Exception as frame_err:
                print(f"Error parsing frame: {frame_err}")
                continue

        print(f"Successfully parsed {len(frames)} frames")
        return frames

    except Exception as e:
        print(f"Error parsing XML file: {e}")
        import traceback
        traceback.print_exc()
        return {}

def create_frame_to_image_mapping(image_folder):
    """
    Create a mapping from frame IDs to image paths.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        frame_to_image (dict): Mapping from frame IDs to image paths.
    """
    frame_to_image = {}

    if not os.path.exists(image_folder):
        print(f"Warning: Image folder {image_folder} does not exist")
        return frame_to_image

    for image_name in os.listdir(image_folder):
        if not any(image_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            continue

        # Try various patterns to extract frame ID
        # Pattern 1: frame_000123.jpg/png
        match = re.search(r'frame_0*(\d+)', image_name.lower())
        if match:
            frame_id = int(match.group(1))
            frame_to_image[frame_id] = os.path.join(image_folder, image_name)
            continue

        # Pattern 2: 000123.jpg
        match = re.search(r'^0*(\d+)', image_name)
        if match:
            frame_id = int(match.group(1))
            frame_to_image[frame_id] = os.path.join(image_folder, image_name)
            continue

        # Pattern 3: Any number in the filename
        match = re.search(r'(\d+)', image_name)
        if match:
            frame_id = int(match.group(1))
            frame_to_image[frame_id] = os.path.join(image_folder, image_name)

    print(f"Found {len(frame_to_image)} images with extractable frame IDs")
    return frame_to_image

class GESCAMCustomDataset(Dataset):
    """
    Dataset class for GESCAM (Gaze Estimation based Synthetic Classroom Attention Measurement)
    Customized for the specific annotation format
    """
    def __init__(self, xml_path, image_folder, transform=None, head_transform=None,
                 input_size=224, output_size=64, test=False):
        """
        Args:
            xml_path (str): Path to the XML annotation file
            image_folder (str): Path to the folder containing images
            transform: Transformations to apply to the scene image
            head_transform: Transformations to apply to the head crop
            input_size: Input image size for the model
            output_size: Output heatmap size
            test: Whether this is a test dataset
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
        Match a person bounding box to the corresponding line of sight polyline

        Args:
            person_box: Dictionary containing person bounding box
            polylines: List of polyline dictionaries for the frame

        Returns:
            target_point: (x,y) tuple of gaze target or None if no match
            has_target: Boolean indicating if a match was found
        """
        # Find polylines labeled as "line of sight"
        sight_lines = [p for p in polylines if p["label"].lower() == "line of sight"]

        if not sight_lines:
            return None, False

        # Calculate person box center
        person_center_x = (person_box["xtl"] + person_box["xbr"]) / 2
        person_center_y = (person_box["ytl"] + person_box["ybr"]) / 2
        person_width = person_box["xbr"] - person_box["xtl"]

        # Find closest matching sight line
        best_match = None
        best_distance = float('inf')

        for polyline in sight_lines:
            points_str = polyline["points"]
            try:
                # Parse points from string format "x1,y1;x2,y2;..."
                points = [tuple(map(float, point.split(","))) for point in points_str.split(";")]

                if len(points) >= 2:  # Need at least start and end point
                    start_x, start_y = points[0]
                    end_x, end_y = points[-1]

                    # Calculate distance from polyline start to person center
                    distance = np.sqrt((start_x - person_center_x)**2 + (start_y - person_center_y)**2)

                    # Check if this is a good match (close to person center)
                    if distance < best_distance and distance < person_width * 1.5:
                        best_distance = distance
                        best_match = (end_x, end_y)  # Use end point as gaze target
            except Exception as e:
                # Print details for debugging
                print(f"Error parsing polyline points: {e}, points_str: {points_str}")
                continue

        return best_match, best_match is not None

    def _create_samples(self):
        """
        Create dataset samples from parsed frames

        Returns:
            samples: List of sample dictionaries
        """
        samples = []
        frames_with_persons = 0
        frames_with_sight_lines = 0

        for frame_id, frame_data in self.frames.items():
            # Skip frames without matching images
            if frame_id not in self.frame_to_image:
                continue

            image_path = self.frame_to_image[frame_id]
            width, height = frame_data["width"], frame_data["height"]

            # Extract object masks for this frame
            object_masks = extract_object_masks_from_annotations(frame_data, width, height)

            # Check if there are person boxes in this frame
            person_boxes = [box for box in frame_data["boxes"] if "person" in box["label"].lower()]
            if person_boxes:
                frames_with_persons += 1

            # Check if there are line of sight polylines
            sight_lines = [p for p in frame_data["polylines"] if p["label"].lower() == "line of sight"]
            if sight_lines:
                frames_with_sight_lines += 1

            # Process each person box
            for person_box in person_boxes:
                # Find matching sight line
                gaze_target, has_target = self._match_person_to_sight_line(person_box, frame_data["polylines"])

                # Create sample
                sample = {
                    "frame_id": frame_id,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "head_bbox": [person_box["xtl"], person_box["ytl"], person_box["xbr"], person_box["ybr"]],
                    "gaze_target": gaze_target,
                    "in_frame": has_target,
                    "object_masks": object_masks  # Add object masks
                }

                samples.append(sample)

        print(f"Statistics: {frames_with_persons} frames with person boxes, {frames_with_sight_lines} frames with sight lines")
        return samples

    def _create_head_position_channel(self, head_bbox, width, height):
        """
        Create a binary mask for head position
        """
        x1, y1, x2, y2 = head_bbox
        head_mask = torch.zeros(height, width)
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(width, x2)), int(min(height, y2))
        head_mask[y1:y2, x1:x2] = 1.0
        return head_mask

    def _create_gaze_heatmap(self, gaze_target, width, height):
        """
        Create a Gaussian heatmap at the gaze point
        """
        if not gaze_target:
            return torch.zeros(self.output_size, self.output_size)

        x, y = gaze_target

        # Scale coordinates to output size
        x = x * self.output_size / width
        y = y * self.output_size / height

        # Create meshgrid
        Y, X = torch.meshgrid(torch.arange(self.output_size), torch.arange(self.output_size), indexing='ij')

        # Create Gaussian heatmap
        sigma = 3.0
        heatmap = torch.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))

        return heatmap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        try:
            img = Image.open(sample["image_path"]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a placeholder if image can't be loaded
            img = Image.new('RGB', (self.input_size, self.input_size), color='gray')

        width, height = sample["width"], sample["height"]

        # Extract head crop
        head_bbox = sample["head_bbox"]
        x1, y1, x2, y2 = head_bbox

        # Ensure bbox is within image bounds
        x1 = max(0, min(width-1, x1))
        y1 = max(0, min(height-1, y1))
        x2 = max(x1+1, min(width, x2))
        y2 = max(y1+1, min(height, y2))

        try:
            head_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        except Exception as e:
            print(f"Error cropping head: {e}, bbox: {head_bbox}, image size: {img.size}")
            head_img = Image.new('RGB', (100, 100), color='gray')

        # Create head position channel
        head_pos = self._create_head_position_channel(head_bbox, width, height)

        # Get object masks
        object_masks = torch.from_numpy(sample["object_masks"])

        # Create gaze heatmap
        if sample["in_frame"] and sample["gaze_target"]:
            gaze_target = sample["gaze_target"]
            gaze_heatmap = self._create_gaze_heatmap(gaze_target, width, height)

            # Calculate gaze vector (from head center to gaze point)
            head_center_x = (x1 + x2) / 2 / width
            head_center_y = (y1 + y2) / 2 / height
            gaze_x = gaze_target[0] / width
            gaze_y = gaze_target[1] / height
            gaze_vector = torch.tensor([gaze_x - head_center_x, gaze_y - head_center_y])
        else:
            gaze_heatmap = torch.zeros(self.output_size, self.output_size)
            gaze_vector = torch.tensor([0.0, 0.0])  # Default for out-of-frame

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        if self.head_transform:
            head_img = self.head_transform(head_img)

        # Resize head position to match input size
        head_pos = head_pos.unsqueeze(0)
        head_pos = F.interpolate(head_pos.unsqueeze(0), size=(self.input_size, self.input_size),
                                 mode='nearest').squeeze(0)

        # Resize object masks to match input size
        object_masks = F.interpolate(object_masks.unsqueeze(0),
                                     size=(self.input_size, self.input_size),
                                     mode='nearest').squeeze(0)

        in_frame = torch.tensor([float(sample["in_frame"])])

        # For compatibility with existing code
        object_label = torch.tensor([0])  # placeholder

        # Instead of returning frame_id as last element (which might cause issues with batching),
        # return a metadata dictionary alongside the tensors
        metadata = {
            "frame_id": sample["frame_id"],
            "image_path": sample["image_path"],
            "head_bbox": sample["head_bbox"],
            "original_size": (width, height)
        }

        return img, head_img, head_pos, gaze_heatmap, in_frame, object_label, gaze_vector, object_masks, metadata

def get_transforms(input_size=224, augment=True):
    """
    Get data transformations for training and validation
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])

    return transform


def visualize_sample_with_objects(sample, save_path=None):
    """
    Visualize a dataset sample with object masks

    Args:
        sample: Tuple of tensors from dataset __getitem__
        save_path: Path to save visualization (if None, displays inline)
    """
    img, head_img, head_pos, gaze_heatmap, in_frame, object_label, gaze_vector, object_masks, metadata = sample

    # Denormalize image
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

    # Create figure
    plt.figure(figsize=(15, 12))

    # Create a 3x3 grid
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

    # Display select object mask channels
    category_names = ["People", "Boards", "Books", "Monitors", "Phones",
                     "Tables", "Water Disp.", "Mugs", "Lamps", "Other1", "Other2"]

    # Display first few object masks
    for i in range(min(4, object_masks.shape[0])):
        plt.subplot(3, 3, 4 + i)
        plt.imshow(object_masks[i].numpy(), cmap='viridis')
        plt.title(f"Object: {category_names[i]}")
        plt.axis('off')

    # Gaze heatmap and visualizations
    plt.subplot(3, 3, 8)
    plt.imshow(gaze_heatmap.numpy(), cmap='jet')
    plt.title(f"Gaze Heatmap (In-frame: {bool(in_frame.item())})")
    plt.axis('off')

    plt.subplot(3, 3, 9)
    # Original image with heatmap overlay
    plt.imshow(img_vis)

    # Resize heatmap to match image size for overlay
    heatmap_vis = gaze_heatmap.numpy()
    heatmap_vis = cv2.resize(heatmap_vis, (img_vis.shape[1], img_vis.shape[0]))

    # Only show heatmap if gaze is in frame
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
    Test the dataset with visualization

    Args:
        xml_path: Path to the XML annotation file
        image_folder: Path to the folder with images
    """
    # Create transforms
    transform = get_transforms(augment=False)

    # Create dataset
    dataset = GESCAMCustomDataset(
        xml_path=xml_path,
        image_folder=image_folder,
        transform=transform
    )

    # Check dataset size
    print(f"\nDataset contains {len(dataset)} samples")

    # If dataset has samples, visualize some
    if len(dataset) > 0:
        print("\nVisualizing samples:")
        num_samples = min(3, len(dataset))
        for i in range(num_samples):
            # Get a sample
            sample_idx = i
            sample = dataset[sample_idx]

            # Visualize
            save_path = f"sample_with_objects_{i}.png"
            visualize_sample_with_objects(sample, save_path)
            print(f"Sample {i} visualization saved to {save_path}")
    else:
        print("No samples to visualize!")

    return dataset
