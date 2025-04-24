from mtcnn import MTCNN
from mtcnn.utils.images import load_image

face = 'data/faceimage.jpeg'
face = 'data/test.png'

# Create a detector instance
detector = MTCNN(device="CPU:0")

# Load an image
image = load_image(face)

# Detect faces in the image
result = detector.detect_faces(image)
print(len(result))

# Display the result
print(result)