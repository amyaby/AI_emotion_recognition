import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("/home/im_ane/emotion_recognition_project/models/emotion_model.h5")

# Load and preprocess the image
image_path = "/home/im_ane/emotion_recognition_project/data/test_images/ana.jpg"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
image = cv2.resize(image, (48, 48))  # Resize to 48x48 (common input size for emotion models)
image = np.expand_dims(image, axis=0)  # Add batch dimension
image = np.expand_dims(image, axis=-1)  # Add channel dimension

# Predict emotion
emotion_prediction = model.predict(image)
emotion_label = np.argmax(emotion_prediction)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
print(f"Predicted Emotion: {emotion_labels[emotion_label]}")

