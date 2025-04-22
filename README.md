# MS-GESCAM

python gescam_demo.py --model path/to/your/model.pt --image path/to/image.jpg --output ./results

python gescam_inference.py --model path/to/your/model.pt --image path/to/image.jpg --head_bbox 0.4 0.4 0.6 0.6 --output ./results

python gescam_demo.py --model path/to/your/model.pt --video path/to/video.mp4 --output ./results

python dnn_face_detector.py --image data/faceimage.jpeg --output results/dnn_detection.png

python dnn_face_detector.py --image data/faceimage.jpeg --confidence 0.3 --output results/lower_confidence.png