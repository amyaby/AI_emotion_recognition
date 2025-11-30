import cv2
from deepface import DeepFace
import sys
import os

def recognize_emotion_from_webcam():
    """
    Reconnaît les émotions en temps réel à partir de la webcam en utilisant DeepFace.
    """
    # DeepFace.stream() est la fonction dédiée pour le flux vidéo en temps réel
    # Elle gère la détection faciale, l'analyse d'émotion et l'affichage.
    # Le paramètre source=0 indique la webcam par défaut.
    # Le paramètre actions=['emotion'] indique de n'analyser que l'émotion.
    
    print("Démarrage de la reconnaissance d'émotions par webcam (DeepFace). Appuyez sur 'q' pour quitter la fenêtre.")
    
    try:
        DeepFace.stream(db_path=".", model_name="Emotion", detector_backend="opencv", enable_face_analysis=True, source=0)
    except Exception as e:
        print(f"Une erreur est survenue lors du démarrage du flux DeepFace: {e}")
        print("Veuillez vous assurer que votre webcam est correctement connectée et accessible.")

if __name__ == "__main__":
    recognize_emotion_from_webcam()
