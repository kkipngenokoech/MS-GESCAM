# endpoints.py
import io
import cv2
import numpy as np
import base64
import os
import uuid
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)
uploads = {}  # Storage for processed frames

def image_to_frame(image_data):
    image_stream = io.BytesIO(image_data)
    image = Image.open(image_stream)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return [frame]

def video_to_frames(video_data):
    video_stream = io.BytesIO(video_data)
    temp_path = f"/tmp/upload_{uuid.uuid4()}.mp4"
    with open(temp_path, 'wb') as f:
        f.write(video_data)
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.remove(temp_path)
        raise ValueError("Failed to open video")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    os.remove(temp_path)
    return frames

def process_upload(file_data, content_type, filename):
    if 'image' in content_type:
        ext = '.png' if 'png' in content_type else '.jpg'
        frames = image_to_frame(file_data)
        # For images, create a temporary video with a single frame
        temp_dir = "/tmp/uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = os.path.splitext(filename)[0] + ".mp4"  # Convert image to video
        temp_path = os.path.join(temp_dir, temp_filename)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_path, fourcc, 1, (width, height))
        video_writer.write(cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        video_writer.release()
    elif 'video' in content_type:
        ext = '.mp4'
        temp_dir = "/tmp/uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)  # Use original filename
        with open(temp_path, 'wb') as f:
            f.write(file_data)
        frames = video_to_frames(file_data)
    else:
        raise ValueError("Unsupported content type")

    try:
        # Call AnalysisMain with the temporary video path
        from test import AnalysisMain
        front_end = AnalysisMain(temp_path)
        if front_end is None:
            raise ValueError("AnalysisMain returned None, likely due to processing failure")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Optionally clean up the temp directory if empty
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

    # Format the output to match the frontend's expectations
    result = {"frame_data": []}
    if not isinstance(front_end, dict):
        raise ValueError(f"Invalid front_end type: {type(front_end)}, expected dict")

    # Convert front_end format to frame_data list
    for frame_key, frame_data in front_end.items():
        if not frame_key.startswith("Frame "):
            continue  # Skip non-frame keys
        try:
            frame_idx = int(frame_key.replace("Frame ", ""))
        except ValueError:
            continue  # Skip invalid frame keys

        # Convert person data to individuals list
        individuals = []
        for person_key, person_data in frame_data.items():
            if person_key.startswith("person "):
                try:
                    person_idx = int(person_key.replace("person ", ""))
                    individuals.append({
                        "person_idx": person_idx,
                        "attention_score": person_data.get("score", 0.0),
                        "looking_at": person_data.get("looking_at", "Unknown")
                    })
                except ValueError:
                    continue  # Skip invalid person keys

        # Convert frame to base64 for storage
        if frame_idx < len(frames):
            _, buffer = cv2.imencode('.jpg', frames[frame_idx])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            frame_base64 = ""

        # Format attention data
        attention_data = {
            "frame_stats": {
                "mean_attention": frame_data.get("mean_attention", 0.0),
                "most_attended_object": frame_data.get("most_attended_object", ""),
                "timestamp": frame_idx
            },
            "individual_scores": individuals
        }

        # Use heatmap path from front_end
        heatmap_data = frame_data.get("heatmap", "")

        # Generate engagement description
        engagement_data = "Engagement level: " + ("High" if frame_data.get("mean_attention", 0.0) > 0.5 else "Low")

        frame_result = {
            "frame_id": frame_idx,
            "attention_data": attention_data,
            "original_frame": frame_base64,
            "visualizations": {
                "heatmap": heatmap_data,
                "engagement": engagement_data
            }
        }
        result["frame_data"].append(frame_result)

    if not result["frame_data"]:
        raise ValueError("No valid frame data found in front_end")

    return result


# Endpoint: Upload image or video
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    print("Received request at /upload_frame")
    if 'file' not in request.files:
        print("Error: No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        print("Error: No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        print(f"Processing file: {file.filename}, content_type: {file.content_type}")
        file_data = file.read()
        upload_id = str(uuid.uuid4())
        analysis_result = process_upload(file_data, file.content_type, file.filename)
        uploads[upload_id] = analysis_result
        response = {"upload_id": upload_id, "frame_count": len(analysis_result["frame_data"])}
        print(f"Upload successful, response: {response}")
        return jsonify(response), 200
    except Exception as e:
        print(f"Error in /upload_frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Endpoint: Get original frame for Raw Feed
@app.route('/get_frame', methods=['POST'])
def get_frame():
    print("Received request at /get_frame")
    data = request.get_json()
    if not data or 'upload_id' not in data or 'frame_idx' not in data:
        print("Error: Missing upload_id or frame_idx")
        return jsonify({"error": "Missing upload_id or frame_idx"}), 400

    upload_id = data['upload_id']
    frame_idx = data['frame_idx']

    if upload_id not in uploads:
        print(f"Error: Upload ID {upload_id} not found")
        return jsonify({"error": "Upload ID not found"}), 404

    analysis = uploads[upload_id]
    frame_data = next((frame for frame in analysis["frame_data"] if frame["frame_id"] == frame_idx), None)
    if frame_data is None:
        print(f"Error: Frame {frame_idx} not found for upload_id {upload_id}")
        return jsonify({"error": "Frame not found"}), 404

    # Return the original frame as base64
    response = {
        "frame_idx": frame_idx,
        "original_frame": frame_data["original_frame"]
    }
    print(f"Returning frame: {response}")
    return jsonify(response)

# Endpoint: Analyze a specific frame
@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    print("Received request at /analyze_frame")
    data = request.get_json()
    if not data or 'upload_id' not in data or 'frame_idx' not in data:
        print("Error: Missing upload_id or frame_idx")
        return jsonify({"error": "Missing upload_id or frame_idx"}), 400

    upload_id = data['upload_id']
    frame_idx = data['frame_idx']

    if upload_id not in uploads:
        print(f"Error: Upload ID {upload_id} not found")
        return jsonify({"error": "Upload ID not found"}), 404

    analysis = uploads[upload_id]
    frame_data = next((frame for frame in analysis["frame_data"] if frame["frame_id"] == frame_idx), None)
    if frame_data is None:
        print(f"Error: Frame {frame_idx} not found for upload_id {upload_id}")
        return jsonify({"error": "Frame not found"}), 404

    # Return the attention data for this frame
    response = frame_data["attention_data"]
    print(f"Returning attention data: {response}")
    return jsonify(response)

# Endpoint: Analyze temporal data (trend across frames)
@app.route('/analyze_temporal', methods=['POST'])
def analyze_temporal():
    print("Received request at /analyze_temporal")
    data = request.get_json()
    if not data or 'upload_id' not in data:
        print("Error: Missing upload_id")
        return jsonify({"error": "Missing upload_id"}), 400

    upload_id = data['upload_id']
    if upload_id not in uploads:
        print(f"Error: Upload ID {upload_id} not found")
        return jsonify({"error": "Upload ID not found"}), 404

    analysis = uploads[upload_id]
    trend = [
        {
            "frame_idx": frame["frame_id"],
            "mean_attention": frame["attention_data"]["frame_stats"]["mean_attention"],
            "timestamp": frame["attention_data"]["frame_stats"]["timestamp"]
        }
        for frame in analysis["frame_data"]
    ]
    response = {"upload_id": upload_id, "trend": trend}
    print(f"Returning temporal analysis: {response}")
    return jsonify(response)

# Endpoint: Visualize a specific frame (Heatmap or Engagement)
@app.route('/visualize_frame', methods=['POST'])
def visualize_frame():
    print("Received request at /visualize_frame")
    data = request.get_json()
    if not data or 'upload_id' not in data or 'frame_idx' not in data or 'vis_type' not in data:
        print("Error: Missing upload_id, frame_idx, or vis_type")
        return jsonify({"error": "Missing upload_id, frame_idx, or vis_type"}), 400

    upload_id = data['upload_id']
    frame_idx = data['frame_idx']
    vis_type = data['vis_type']

    if upload_id not in uploads:
        print(f"Error: Upload ID {upload_id} not found")
        return jsonify({"error": "Upload ID not found"}), 404

    analysis = uploads[upload_id]
    frame_data = next((frame for frame in analysis["frame_data"] if frame["frame_id"] == frame_idx), None)
    if frame_data is None:
        print(f"Error: Frame {frame_idx} not found for upload_id {upload_id}")
        return jsonify({"error": "Frame not found"}), 404

    if vis_type not in frame_data["visualizations"]:
        print(f"Error: Invalid vis_type {vis_type}")
        return jsonify({"error": "Invalid vis_type"}), 400

    response = {
        "frame_idx": frame_idx,
        "vis_type": vis_type,
        "vis_image": frame_data["visualizations"][vis_type]
    }
    print(f"Returning visualization: {response}")
    return jsonify(response)

if __name__ == "__main__":
    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)