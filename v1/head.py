from ultralytics import YOLO
from v1.face_detector import face

# Load the head detection model
model = YOLO('yolov8n-head.pt')

# Run inference on an image
results = model(face)

# Extract bounding boxes
bboxes = []
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get box coordinates in (x1, y1, x2, y2) format
        confidence = box.conf[0].item()         # Get confidence score
        bboxes.append([x1, y1, x2, y2, confidence])

print(bboxes)