import cv2
from deepface import DeepFace
import sys
import os

def recognize_emotion_from_image(image_path):
    """
    Reconnaît les émotions sur les visages d'une image statique en utilisant DeepFace.
    """
    if not os.path.exists(image_path):
        print(f"Erreur: Le fichier image n'existe pas à l'emplacement: {image_path}")
        return

    try:
        # Analyse de l'image avec DeepFace
        # actions=['emotion'] se concentre uniquement sur la reconnaissance d'émotions
        # enforce_detection=False permet de traiter les images même si la détection faciale est difficile
        results = DeepFace.analyze(
            img_path=image_path, 
            actions=['emotion'], 
            enforce_detection=False
        )
        
        # Charger l'image pour l'affichage
        img = cv2.imread(image_path)
        
        if not isinstance(results, list):
            results = [results]

        for result in results:
            emotion = result['dominant_emotion']
            region = result['region']
            
            # Coordonnées du visage
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Dessiner un rectangle et afficher l'émotion
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            print(f"Visage détecté à ({x}, {y}) avec l'émotion dominante: {emotion}")

        # Enregistrer l'image résultante
        output_path = "/home/im_ane/AI_emotion_recognition/results/result_emotion_image_deepface.jpg"
        cv2.imwrite(output_path, img)
        print(f"Image traitée enregistrée sous: {output_path}")
        return output_path

    except Exception as e:
        print(f"Une erreur est survenue lors de l'analyse DeepFace: {e}")
        print("DeepFace peut nécessiter le téléchargement de modèles la première fois. Veuillez réessayer.")
        return

if __name__ == "__main__":
    # Vérifier si un argument de chemin d'image est fourni
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
    else:
        # Chemin par défaut pour le STEP (image fournie par l'utilisateur)
        input_image_path = "/home/im_ane/AI_emotion_recognition/data/test_images/2.jpg"
        print(f"Aucun chemin d'image fourni. Utilisation de l'image par défaut: {input_image_path}")

    recognize_emotion_from_image(input_image_path)
