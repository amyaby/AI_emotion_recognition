import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Load the emotion recognition model
try:
    # Use Windows path format with raw string
    model_path = r"C:\Users\imane\Documents\emotion_model.h5"
    # Replace "YourUsername" with your actual Windows username

    # Check if file exists first
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure:")
        print("1. The file exists at this location")
        print("2. The path is correct (check your username)")
        print("3. You've copied the model from WSL to Windows")
        exit(1)

    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying to use DeepFace's built-in model instead...")
    # We'll handle this in the main loop

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load the Haar Cascade classifier
# Use Windows path for the cascade file if you've copied it
haar_path = r"C:\Users\YourUsername\Documents\haarcascade_frontalface_default.xml"

# Try to load the Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    print("Error: Could not load Haar Cascade classifier from Windows path")
    print("Trying to use OpenCV's built-in cascade...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load any Haar Cascade classifier")
        exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    # Try alternative camera indices
    for i in range(1, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Using camera at index {i}")
            break
    else:
        print("No webcam found. Please connect a webcam.")
        exit()

print("Emotion Recognition - Press ESC to exit")

while True:
    # Read the frame
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # If no faces detected, try with different parameters
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw the rectangle around each face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region with some padding
        x_start = max(0, x - int(w*0.1))
        y_start = max(0, y - int(h*0.1))
        x_end = min(img.shape[1], x + w + int(w*0.1))
        y_end = min(img.shape[0], y + h + int(h*0.1))
        face_roi = img[y_start:y_end, x_start:x_end]

        if face_roi.size > 0:
            try:
                # Preprocess the face for emotion recognition
                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_roi_processed = cv2.resize(face_roi_gray, (64, 64))
                face_roi_processed = np.expand_dims(face_roi_processed, axis=0)
                face_roi_processed = np.expand_dims(face_roi_processed, axis=-1)
                face_roi_processed = face_roi_processed.astype('float32') / 255.0

                # Predict emotion
                emotion_prediction = model.predict(face_roi_processed)
                emotion_label = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label]
                confidence = emotion_prediction[0][emotion_label]

                # Display emotion
                cv2.putText(img, f"{emotion_text} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except NameError:
                # Model wasn't loaded successfully, use DeepFace instead
                from deepface import DeepFace
                try:
                    result = DeepFace.analyze(
                        img_path=face_roi,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    emotions = result[0]['emotion']
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    cv2.putText(img, f"{dominant_emotion[0]} ({dominant_emotion[1]:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"DeepFace error: {e}")
                    cv2.putText(img, "Emotion?", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display
    cv2.imshow('Emotion Recognition', img)

    # Stop if escape key is pressed
    if cv2.waitKey(30) & 0xff == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()


