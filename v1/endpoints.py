# -*- coding: utf-8 -*-
"""endpoints.py

Flask API for MS-GESCAM with ngrok tunneling
"""

import io
import cv2
import numpy as np
import base64
import os
import uuid
from PIL import Image
import subprocess
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf

# Dummy storage for uploaded frames
Uploads = {}

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
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
    cap.release()
    os.remove(temp_path)
    print(f"Extracted {frame_count} frames from video")
    return frames

class Backend_send_frames:
    def __init__(self):
        pass

    def send_frames(self, virtual_path):
        return None

def process_upload(file_data, content_type):
    if 'image' in content_type:
        ext = '.png' if 'png' in content_type else '.jpg'
        frames = image_to_frame(file_data)
    elif 'video' in content_type:
        ext = '.mp4'
        frames = video_to_frames(file_data)
    else:
        raise ValueError("Unsupported content type")

    virtual_path = f"/tmp/upload_{uuid.uuid4()}{ext}"
    with open(virtual_path, 'wb') as f:
        f.write(file_data)

    backend = Backend_send_frames()
    result = {"frame_data": []}
    frame_indices = [0, 1, 2, 8]
    test_data = {
        0: {
            "mean_attention": 0.1252141764476059,
            "most_attended_object": "Desk/Table",
            "heatmap": "output/attention_test_results/frame_0_attention.png",
            "individuals": [
                {"person_idx": i, "score": 0.1 + i*0.01, "looking_at": "Desk/Table" if i % 2 == 0 else "Person/Teacher"}
                for i in range(13)
            ]
        },
        1: {
            "mean_attention": 0.1254738093868792,
            "most_attended_object": "Desk/Table",
            "heatmap": "output/attention_test_results/frame_1_attention.png",
            "individuals": [
                {"person_idx": i, "score": 0.1 + i*0.01, "looking_at": "Desk/Table" if i % 2 == 0 else "Person/Teacher"}
                for i in range(13)
            ]
        },
        2: {
            "mean_attention": 0.12396571152511779,
            "most_attended_object": "Desk/Table",
            "heatmap": "output/attention_test_results/frame_2_attention.png",
            "individuals": [
                {"person_idx": i, "score": 0.1 + i*0.01, "looking_at": "Desk/Table" if i % 2 == 0 else "Person/Teacher"}
                for i in range(13)
            ]
        },
        8: {
            "mean_attention": 0.12139347740718362,
            "most_attended_object": "Desk/Table",
            "heatmap": "output/attention_test_results/frame_8_attention.png",
            "individuals": [
                {"person_idx": i, "score": 0.1 + i*0.01, "looking_at": "Desk/Table" if i % 2 == 0 else "Person/Teacher"}
                for i in range(13)
            ]
        }
    }

    for idx, frame in enumerate(frames):
        frame_idx = frame_indices[idx % len(frame_indices)]
        frame_data = test_data[frame_idx]
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        individual_scores = [
            {
                "person_idx": person["person_idx"],
                "attention_score": person["score"],
                "looking_at": person["looking_at"]
            }
            for person in frame_data["individuals"]
        ]

        attention_data = {
            "frame_stats": {
                "mean_attention": frame_data["mean_attention"],
                "most_attended_object": frame_data["most_attended_object"],
                "timestamp": idx
            },
            "individual_scores": individual_scores
        }

        heatmap_data = frame_data["heatmap"]
        engagement_data = f"Engagement level: {'High' if frame_data['mean_attention'] > 0.5 else 'Low'}"

        frame_result = {
            "frame_id": idx,
            "attention_data": attention_data,
            "original_frame": frame_base64,
            "visualizations": {
                "heatmap": heatmap_data,
                "engagement": engagement_data
            }
        }
        result["frame_data"].append(frame_result)

    if os.path.exists(virtual_path):
        os.remove(virtual_path)

    return result

# Install required packages
try:
    subprocess.run(["pip", "install", "flask", "pyngrok", "opencv-python", "pillow", "numpy"], check=True)
except subprocess.CalledProcessError:
    print("Failed to install required packages. Ensure pip is working correctly.")

app = Flask(__name__)

# Flask Endpoints
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
        analysis_result = process_upload(file_data, file.content_type)

        Uploads[upload_id] = analysis_result
        response = {"upload_id": upload_id, "frame_count": len(analysis_result["frame_data"])}
        print(f"Upload successful, response: {response}")
        return jsonify(response), 200
    except Exception as e:
        print(f"Error in /upload_frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_frame', methods=['POST'])
def get_frame():
    print("Received request at /get_frame")
    data = request.get_json()
    if not data or 'upload_id' not in data or 'frame_idx' not in data:
        print("Error: Missing upload_id or frame_idx")
        return jsonify({"error": "Missing upload_id or frame_idx"}), 400

    upload_id = data['upload_id']
    frame_idx = data['frame_idx']

    if upload_id not in Uploads:
        print(f"Error: Upload ID {upload_id} not found")
        return jsonify({"error": "Upload ID not found"}), 404

    analysis = Uploads[upload_id]
    frame_data = next((frame for frame in analysis["frame_data"] if frame["frame_id"] == frame_idx), None)
    if frame_data is None:
        print(f"Error: Frame {frame_idx} not found for upload_id {upload_id}")
        return jsonify({"error": "Frame not found"}), 404

    response = {
        "frame_idx": frame_idx,
        "original_frame": frame_data["original_frame"]
    }
    print(f"Returning frame: {response}")
    return jsonify(response)

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    print("Received request at /analyze_frame")
    data = request.get_json()
    if not data or 'upload_id' not in data or 'frame_idx' not in data:
        print("Error: Missing upload_id or frame_idx")
        return jsonify({"error": "Missing upload_id or frame_idx"}), 400

    upload_id = data['upload_id']
    frame_idx = data['frame_idx']

    if upload_id not in Uploads:
        print(f"Error: Upload ID {upload_id} not found")
        return jsonify({"error": "Upload ID not found"}), 404

    analysis = Uploads[upload_id]
    frame_data = next((frame for frame in analysis["frame_data"] if frame["frame_id"] == frame_idx), None)
    if frame_data is None:
        print(f"Error: Frame {frame_idx} not found for upload_id {upload_id}")
        return jsonify({"error": "Frame not found"}), 404

    response = frame_data["attention_data"]
    print(f"Returning attention data: {response}")
    return jsonify(response)

@app.route('/analyze_temporal', methods=['POST'])
def analyze_temporal():
    print("Received request at /analyze_temporal")
    data = request.get_json()
    if not data or 'upload_id' not in data:
        print("Error: Missing upload_id")
        return jsonify({"error": "Missing upload_id"}), 400

    upload_id = data['upload_id']
    if upload_id not in Uploads:
        print(f"Error: Upload ID {upload_id} not found")
        return jsonify({"error": "Upload ID not found"}), 404

    analysis = Uploads[upload_id]
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

    if upload_id not in Uploads:
        print(f"Error: Upload ID {upload_id} not found")
        return jsonify({"error": "Upload ID not found"}), 404

    analysis = Uploads[upload_id]
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
    # Use the Flask ngrok URL from 'ngrok start --all'
    public_url = "https://ef1d-41-186-78-169.ngrok-free.app"
    print(f" * Flask ngrok tunnel: {public_url}")

    # Start Flask server
    app.run(host="0.0.0.0", port=5000)