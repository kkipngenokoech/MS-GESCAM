from DatasetModule import GESCAMCustomDataset, get_transforms, test_dataset, visualize_sample_with_objects
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
from tqdm import tqdm
from analysis import test_attention_scoring
import json
import torch
from GESCAMDATA.videoloader import test_video_loader

# Define the paths

_xml_path = "GESCAMDATA/test_subset/task_classroom_11_video-01_final/annotations.xml"
_folder = "GESCAMDATA/test_subset/task_classroom_11_video-01_final/images"
folder = "GESCAMDATA/test_subset/task_classroom_11_video-01_final/demo"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
# Create the output directory if it doesn't exist




def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj





def apiEndpoint(folder, xml_path=None):
    """
    This function is a placeholder for the API endpoint that would be used to
    process the data. It currently does not perform any operations.'
    """
    
    
    model_path = "best_model.pt"  # Path to your trained model
    test_output_dir = os.path.join(output_dir, "attention_test_results")
    xml_path = _xml_path
    dataset_path = xml_path


    # Run the test
    _, _, front_end = test_attention_scoring(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=test_output_dir,
        num_frames=300
    )
    results_path = os.path.join(test_output_dir, "test_results.json")
    temporal_path = os.path.join(test_output_dir, "temporal_data.json")
    
    with open(results_path, 'w') as f:
        json.dump(front_end, f, indent=4, default=to_serializable)

   


    return front_end    # Save the results to a JSON file



def main():
    # Set paths to your data
    xml_path = _xml_path
    image_folder = folder


    # Create transforms
    transform = get_transforms(augment=False)

    print("Creating dataset...")
    # Create dataset with the customized class
    dataset = GESCAMCustomDataset(
        xml_path=xml_path,
        image_folder=image_folder,
        transform=transform
    )

    # If we have samples, create a DataLoader and visualize
    if len(dataset) > 0:
        # Create DataLoader
        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Created DataLoader with batch size {batch_size}")

        # Visualize some samples
        print("Visualizing samples...")
        num_samples = min(5, len(dataset))
        for i in range(num_samples):
            # Choose random sample for variety
            sample_idx = np.random.randint(0, len(dataset))
            sample = dataset[sample_idx]

            # Visualize
            save_path = os.path.join(output_dir, f"sample_{i}.png")
            # Use the new visualization function that handles object masks
            visualize_sample_with_objects(sample, save_path)
            print(f"Sample {i} visualization saved to {save_path}")

        # Create a video visualization
        create_visualization_video(dataset, os.path.join(output_dir, "visualization.mp4"), num_samples=min(30, len(dataset)), fps=2)
    else:
        print("No samples found in the dataset!")

    print("Done!")

def create_visualization_video(dataset, output_video_path, num_samples=20, fps=5):
    """
    Create a video visualizing dataset samples

    Args:
        dataset: Dataset instance
        output_video_path: Path to save the video
        num_samples: Number of samples to include
        fps: Frames per second
    """
    if len(dataset) == 0:
        print("Cannot create video with empty dataset")
        return

    print(f"Creating visualization video with {num_samples} samples...")

    # Create a temporary directory for frames
    temp_dir = "temp_viz_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Get evenly distributed sample indices
    indices = np.linspace(0, len(dataset)-1, num_samples).astype(int)

    # Visualize each sample
    for i, idx in enumerate(tqdm(indices, desc="Generating frames")):
        sample = dataset[idx]

        # Save visualization to temp file
        temp_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        # Use the new visualization function that handles object masks
        visualize_sample_with_objects(sample, temp_path)

    # Get size of the first frame to set video dimensions
    first_frame = cv2.imread(os.path.join(temp_dir, "frame_0000.png"))
    height, width, _ = first_frame.shape  # Fixed the syntax error here

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Add frames to video
    for i in range(len(indices)):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release video writer
    video_writer.release()

    # Clean up temporary files
    for i in range(len(indices)):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        if os.path.exists(frame_path):
            os.remove(frame_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    print(f"Visualization video saved to {output_video_path}")


def AnalysisMain(_video_path):
    # Example usage
    folder, _xml_path = test_video_loader(
        video_path=_video_path,
        data_dir="GESCAMDATA/",
        output_frame_base_dir="GESCAMDATA/video2frames/extraxted_frames/task_classroom_11_video-01_final"
    )
    apiEndpoint(folder, _xml_path)

if __name__ == "__main__":
    # Corrected: Remove trailing comma to ensure video_path is a string
    video_path = "GESCAMDATA/video2frames/videos/task_classroom_11_video-01_final.mp4"
    AnalysisMain(video_path)