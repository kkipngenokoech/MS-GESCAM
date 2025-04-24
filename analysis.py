import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
from tqdm import tqdm
from DatasetModule import GESCAMCustomDataset, get_transforms
from modelArch import MSGESCAMModel, device


def calculate_individual_attention(pred_heatmap, object_masks, context="lecture"):
    """
    Calculate individual attention score based on gaze heatmap and object masks

    Args:
        pred_heatmap: Predicted gaze heatmap (numpy array)
        object_masks: Dictionary or tensor of object masks with class indices
        context: Teaching context (lecture, group_work, individual_work)

    Returns:
        attention_score: Overall attention score (0-1)
        object_attention: Dictionary of attention scores per object category
    """
    # Define object importance based on educational context
    if context == "lecture":
        # During lectures, looking at teacher/board is most important
        importance_weights = {
            0: 0.7,  # person/teacher
            1: 0.9,  # blackboard/whiteboard
            2: 0.6,  # books/notebooks
            3: 0.5,  # monitors/screens
            4: 0.1,  # phones (usually distractions)
            5: 0.3,  # desks/tables
            6: 0.1,  # water dispenser (distraction)
            7: 0.1,  # mugs (distraction)
            8: 0.1,  # lamps (distraction)
            9: 0.1,  # other objects
            10: 0.1  # other objects
        }
    elif context == "group_work":
        # During group work, looking at peers and materials is valuable
        importance_weights = {
            0: 0.8,  # person/peers
            1: 0.5,  # blackboard/whiteboard
            2: 0.7,  # books/notebooks
            3: 0.6,  # monitors/screens
            4: 0.1,  # phones (usually distractions)
            5: 0.4,  # desks/tables
            6: 0.1,  # water dispenser (distraction)
            7: 0.1,  # mugs (distraction)
            8: 0.1,  # lamps (distraction)
            9: 0.1,  # other objects
            10: 0.1  # other objects
        }
    else:  # individual_work
        # During individual work, focus on learning materials
        importance_weights = {
            0: 0.3,  # person/teacher
            1: 0.4,  # blackboard/whiteboard
            2: 0.9,  # books/notebooks
            3: 0.8,  # monitors/screens
            4: 0.1,  # phones (usually distractions)
            5: 0.5,  # desks/tables
            6: 0.1,  # water dispenser (distraction)
            7: 0.1,  # mugs (distraction)
            8: 0.1,  # lamps (distraction)
            9: 0.1,  # other objects
            10: 0.1  # other objects
        }

    # Normalize the heatmap
    heatmap_normalized = pred_heatmap / pred_heatmap.sum() if pred_heatmap.sum() > 0 else pred_heatmap

    # Calculate attention overlap with each object class
    object_attention = {}
    total_weighted_attention = 0
    max_overlap = 0
    max_object = None

    # Process each object category
    for class_idx in range(object_masks.shape[0]):
        # Ensure masks are same size as heatmap
        mask = object_masks[class_idx]
        if mask.shape != heatmap_normalized.shape:
            mask = cv2.resize(mask, (heatmap_normalized.shape[1], heatmap_normalized.shape[0]))

        # Calculate overlap (element-wise multiplication)
        overlap = np.sum(heatmap_normalized * mask)

        # Scale by importance weight
        weighted_overlap = overlap * importance_weights.get(class_idx, 0.1)

        # Track which object has most attention
        if overlap > max_overlap:
            max_overlap = overlap
            max_object = class_idx

        # Store results
        object_attention[class_idx] = {
            'raw_overlap': float(overlap),
            'weighted_score': float(weighted_overlap)
        }

        total_weighted_attention += weighted_overlap

    # Calculate final score (0-1)
    # Normalize to 0-1 range
    attention_score = min(1.0, total_weighted_attention)

    # Add information about primary object of attention
    object_attention['primary_object'] = max_object
    object_attention['primary_overlap'] = float(max_overlap)

    return attention_score, object_attention

def analyze_frame_attention(model, frame_data, device, context="lecture"):
    """
    Analyze attention for all people in a single frame

    Args:
        model: Trained attention model
        frame_data: List of samples for people in the frame
        device: Device to run model on
        context: Teaching context

    Returns:
        frame_attention: Dictionary with individual and aggregated attention data
    """
    individual_scores = []
    individual_heatmaps = []
    primary_targets = {}

    # Dict to track how many people look at each object type
    objects_attention_count = {i: 0 for i in range(11)}  # 11 object categories

    with torch.no_grad():
        for person_idx, sample in enumerate(frame_data):
            # Unpack sample
            scene_img, head_img, head_pos, _, target_in_frame, _, _, object_masks, metadata = sample

            # Skip if no target in frame
            if not target_in_frame.item():
                continue

            # Prepare inputs for model (add batch dimension)
            scene_img = scene_img.unsqueeze(0).to(device)
            head_img = head_img.unsqueeze(0).to(device)
            head_pos = head_pos.unsqueeze(0).to(device)
            object_masks = object_masks.unsqueeze(0).to(device)

            # Forward pass
            pred_heatmap, _ = model(scene_img, head_img, head_pos, object_masks)

            # Convert to numpy for processing
            pred_heatmap_np = pred_heatmap.squeeze().cpu().numpy()
            object_masks_np = object_masks.squeeze().cpu().numpy()

            # Calculate attention score
            attention_score, object_attention = calculate_individual_attention(
                pred_heatmap_np, object_masks_np, context)

            # Store individual results
            individual_scores.append({
                'person_idx': person_idx,
                'attention_score': attention_score,
                'object_attention': object_attention,
                'primary_object': object_attention['primary_object'],
                'head_bbox': metadata['head_bbox']
            })

            individual_heatmaps.append(pred_heatmap_np)

            # Count primary targets
            primary_obj = object_attention['primary_object']
            if primary_obj is not None:
                objects_attention_count[primary_obj] += 1
                primary_targets[primary_obj] = primary_targets.get(primary_obj, 0) + 1

    # Calculate frame-level statistics
    if individual_scores:
        mean_attention = sum(item['attention_score'] for item in individual_scores) / len(individual_scores)

        # Find most commonly attended object
        most_attended_object = max(objects_attention_count.items(), key=lambda x: x[1]) if objects_attention_count else None

        # Calculate percentage of people attending to each object type
        total_people = len(individual_scores)
        attention_distribution = {obj: count/total_people for obj, count in objects_attention_count.items() if count > 0}

        # Create combined attention heatmap
        if individual_heatmaps:
            combined_heatmap = sum(individual_heatmaps)
            # Normalize
            if combined_heatmap.max() > 0:
                combined_heatmap = combined_heatmap / combined_heatmap.max()
        else:
            combined_heatmap = None
    else:
        mean_attention = 0
        most_attended_object = None
        attention_distribution = {}
        combined_heatmap = None

    # Create result dictionary
    frame_attention = {
        'individual_scores': individual_scores,
        'frame_stats': {
            'mean_attention': mean_attention,
            'most_attended_object': most_attended_object,
            'attention_distribution': attention_distribution,
            'total_people': len(individual_scores)
        },
        'combined_heatmap': combined_heatmap
    }

    return frame_attention


def visualize_individual_attention(frame_img, attention_data, object_names=None, save_path=None):
    """
    Visualize individual attention scores in a frame

    Args:
        frame_img: Original frame image (RGB)
        attention_data: Attention data from analyze_frame_attention
        object_names: Dictionary mapping object indices to names
        save_path: Path to save visualization
    """
    if object_names is None:
        object_names = {
            0: "Person/Teacher",
            1: "Board",
            2: "Book/Notebook",
            3: "Monitor/Screen",
            4: "Phone",
            5: "Desk/Table",
            6: "Water Dispenser",
            7: "Mug",
            8: "Lamp",
            9: "Other1",
            10: "Other2"
        }

    # Create figure
    plt.figure(figsize=(15, 10))

    # Show frame with scores
    plt.subplot(1, 2, 1)
    plt.imshow(frame_img)

    # Add bounding boxes and scores
    for person_data in attention_data['individual_scores']:
        x1, y1, x2, y2 = person_data['head_bbox']
        score = person_data['attention_score']

        # Color based on score (green for high, yellow for medium, red for low)
        if score > 0.7:
            color = 'lime'
        elif score > 0.4:
            color = 'yellow'
        else:
            color = 'red'

        # Add rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)

        # Add score text
        primary_obj = object_names.get(person_data['primary_object'], "Unknown")
        plt.text(x1, y1-10, f"Score: {score:.2f}\nLooking at: {primary_obj}",
                color='white', fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7))

    # Show combined heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(frame_img)

    if attention_data['combined_heatmap'] is not None:
        # Resize heatmap to match image
        h, w = frame_img.shape[:2]
        heatmap = cv2.resize(attention_data['combined_heatmap'], (w, h))
        plt.imshow(heatmap, cmap='jet', alpha=0.6)

    # Add frame stats
    stats = attention_data['frame_stats']
    most_obj = object_names.get(stats['most_attended_object'][0], "None") if stats['most_attended_object'] else "None"

    plt.title(f"Class Attention | Mean Score: {stats['mean_attention']:.2f} | " +
             f"Most Attended: {most_obj} ({stats['most_attended_object'][1] if stats['most_attended_object'] else 0} people)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_attention_over_time(temporal_data, object_names=None, save_path=None):
    """
    Visualize attention metrics over time

    Args:
        temporal_data: Output from track_attention_over_time
        object_names: Dictionary mapping object indices to names
        save_path: Path to save visualization
    """
    if object_names is None:
        object_names = {
            0: "Person/Teacher",
            1: "Board",
            2: "Book/Notebook",
            3: "Monitor/Screen",
            4: "Phone",
            5: "Desk/Table",
            6: "Water Dispenser",
            7: "Mug",
            8: "Lamp",
            9: "Other1",
            10: "Other2"
        }

    # Extract time series data
    attention_over_time = temporal_data['temporal_stats']['attention_over_time']
    targets_over_time = temporal_data['temporal_stats']['targets_over_time']
    frame_ids = [frame['frame_id'] for frame in temporal_data['frame_data']]

    # Create figure
    plt.figure(figsize=(15, 8))

    # Plot attention scores over time
    plt.subplot(2, 1, 1)
    plt.plot(frame_ids, attention_over_time, 'b-o', linewidth=2)
    plt.xlabel('Frame ID')
    plt.ylabel('Mean Attention Score')
    plt.title(f'Attention Score Over Time (Stability: {temporal_data["temporal_stats"]["attention_stability"]:.3f})')
    plt.grid(True)

    # Plot attention targets over time
    plt.subplot(2, 1, 2)

    # Convert target indices to names
    target_names = []
    for target in targets_over_time:
        if target and target[0] in object_names:
            target_names.append(object_names[target[0]])
        else:
            target_names.append("None")

    # Create colored markers for each unique target
    unique_targets = set(target_names)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_targets)))
    target_colors = {target: colors[i] for i, target in enumerate(unique_targets)}

    # Plot points colored by target
    for i, (frame, target) in enumerate(zip(frame_ids, target_names)):
        plt.scatter(frame, i % 5, color=target_colors[target], s=100, label=target if i == 0 or target_names[i-1] != target else "")

        # Add text label
        plt.text(frame, i % 5 + 0.3, target, rotation=45, ha='right')

    # Add shift markers
    for shift in temporal_data['temporal_stats']['attention_shifts']:
        plt.axvline(x=shift['frame_id'], color='r', linestyle='--', alpha=0.5)

    plt.xlabel('Frame ID')
    plt.title('Attention Target Shifts Over Time')

    # Remove y-axis as it's just for spacing
    plt.yticks([])

    # Handle legend (only show unique targets)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()









def test_attention_scoring(model_path, dataset_path, output_dir, num_frames=5):
    """
    Test the attention scoring functionality independently

    Args:
        model_path: Path to trained model checkpoint
        dataset_path: Path to dataset (XML annotation file)
        output_dir: Directory to save test results
        num_frames: Number of frames to analyze
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    

    # Load model
    print("Loading model...")
    model = MSGESCAMModel(pretrained=False, output_size=64)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using randomly initialized model for testing")

    model = model.to(device)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    image_folder = os.path.join(os.path.dirname(dataset_path), "demo")
    transform = get_transforms(augment=False)

    dataset = GESCAMCustomDataset(
        xml_path=dataset_path,
        image_folder=image_folder,
        transform=transform
    )

    print(f"Dataset loaded with {len(dataset)} samples")

    # Get unique frame IDs
    all_frame_ids = []
    frame_id_to_samples = {}

    print("Indexing frames...")
    for idx in range(len(dataset)):
        sample = dataset[idx]
        metadata = sample[8]  # Metadata is the 9th element
        frame_id = metadata['frame_id']

        # Store mapping of frame ID to sample indices
        if frame_id not in frame_id_to_samples:
            frame_id_to_samples[frame_id] = []
            all_frame_ids.append(frame_id)

        frame_id_to_samples[frame_id].append(idx)

    # Select random frames
    if len(all_frame_ids) <= num_frames:
        selected_frames = all_frame_ids
    else:
        selected_frames = sorted(np.random.choice(all_frame_ids, num_frames, replace=False))

    print(f"Selected {len(selected_frames)} frames for analysis")

    # Define object class names
    object_names = {
        0: "Person/Teacher",
        1: "Board",
        2: "Book/Notebook",
        3: "Monitor/Screen",
        4: "Phone",
        5: "Desk/Table",
        6: "Water Dispenser",
        7: "Mug",
        8: "Lamp",
        9: "Other1",
        10: "Other2"
    }

    # Process each frame
    results = []

    for frame_idx, frame_id in enumerate(tqdm(selected_frames, desc="Analyzing frames")):
        # Get all samples for this frame
        frame_sample_indices = frame_id_to_samples[frame_id]
        frame_samples = [dataset[idx] for idx in frame_sample_indices]

        # Get one sample to extract the frame image
        first_sample = frame_samples[0]
        scene_img = first_sample[0]

        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_vis = scene_img.clone()
        img_vis = img_vis * std + mean
        img_vis = img_vis.permute(1, 2, 0).numpy()
        img_vis = np.clip(img_vis, 0, 1)

        # Analyze frame attention
        frame_attention = analyze_frame_attention(
            model=model,
            frame_data=frame_samples,
            device=device,
            context="lecture"  # Default context
        )

        # Store result
        results.append({
            'frame_id': frame_id,
            'attention_data': frame_attention
        })

        # Visualize individual attention
        vis_path = os.path.join(output_dir, f"frame_{frame_id}_attention.png")
        visualize_individual_attention(
            frame_img=img_vis,
            attention_data=frame_attention,
            object_names=object_names,
            save_path=vis_path
        )

        print(f"Frame {frame_id} visualization saved to {vis_path}")

        # Print summary of this frame
        print(f"\nFrame {frame_id} Analysis Summary:")
        print(f"- Total people: {frame_attention['frame_stats']['total_people']}")
        print(f"- Mean attention score: {frame_attention['frame_stats']['mean_attention']:.2f}")

        if frame_attention['frame_stats']['most_attended_object']:
            obj_idx, count = frame_attention['frame_stats']['most_attended_object']
            print(f"- Most attended object: {object_names.get(obj_idx, 'Unknown')} ({count} people)")

        print("- Individual scores:")
        for person in frame_attention['individual_scores']:
            primary_obj = object_names.get(person['primary_object'], "Unknown")
            print(f"  Person {person['person_idx']}: Score={person['attention_score']:.2f}, Looking at: {primary_obj}")

    # Create temporal analysis if we have multiple frames
    if len(results) > 1:
        # Convert results to temporal_data format
        temporal_data = {
            'frame_data': results,
            'temporal_stats': {
                'attention_over_time': [r['attention_data']['frame_stats']['mean_attention'] for r in results],
                'targets_over_time': [r['attention_data']['frame_stats']['most_attended_object'] for r in results],
                'attention_stability': np.std([r['attention_data']['frame_stats']['mean_attention'] for r in results]),
                'attention_shifts': []
            }
        }

        # Detect attention shifts
        for i in range(1, len(results)):
            prev = results[i-1]['attention_data']['frame_stats']['most_attended_object']
            curr = results[i]['attention_data']['frame_stats']['most_attended_object']

            # Check if primary target changed
            if prev and curr and prev[0] != curr[0]:
                temporal_data['temporal_stats']['attention_shifts'].append({
                    'frame_id': results[i]['frame_id'],
                    'from_object': prev,
                    'to_object': curr
                })

        # Visualize temporal data
        vis_path = os.path.join(output_dir, "attention_over_time.png")
        visualize_attention_over_time(
            temporal_data=temporal_data,
            object_names=object_names,
            save_path=vis_path
        )

        print(f"\nTemporal analysis visualization saved to {vis_path}")
        print(f"Attention stability: {temporal_data['temporal_stats']['attention_stability']:.3f}")
        print(f"Detected {len(temporal_data['temporal_stats']['attention_shifts'])} major attention shifts")

    print("\nAttention analysis testing complete!")
    return results

