"""
RECONNAISSANCE D'ÉMOTIONS AVEC DEEPFACE - VERSION CORRIGÉE
===========================================================
Reconnaît les émotions sur les visages avec DeepFace
Correction des textes qui se chevauchent quand plusieurs visages

Usage:
    python emotion_recognition_deepface_enhanced.py [image_path]
    
Touches:
    - ESC ou Q : Fermer la fenêtre
    - S : Sauvegarder l'image
"""

import cv2
from deepface import DeepFace
import sys
import os
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

# Dossier de sortie
RESULTS_DIR = "/home/im_ane/AI_emotion_recognition/results"
DEFAULT_IMAGE = "/home/im_ane/AI_emotion_recognition/data/test_images/billie.jpeg"

# Créer le dossier results s'il n'existe pas
os.makedirs(RESULTS_DIR, exist_ok=True)

# Couleurs pour chaque émotion (BGR)
EMOTION_COLORS = {
    'angry': (0, 0, 255),       # Rouge
    'disgust': (0, 128, 128),   # Teal
    'fear': (128, 0, 128),      # Violet
    'happy': (0, 255, 0),       # Vert
    'sad': (255, 0, 0),         # Bleu
    'surprise': (0, 165, 255),  # Orange
    'neutral': (200, 200, 200)  # Gris
}

print("="*70)
print("🎭 RECONNAISSANCE D'ÉMOTIONS AVEC DEEPFACE")
print("="*70)

# ============================================
# FONCTION PRINCIPALE (CORRIGÉE)
# ============================================

def recognize_emotion_from_image(image_path, display=True, save=True):
    """
    Reconnaît les émotions sur les visages d'une image statique
    
    Args:
        image_path (str): Chemin de l'image
        display (bool): Afficher l'image dans une fenêtre
        save (bool): Sauvegarder le résultat
        
    Returns:
        str: Chemin de l'image sauvegardée (ou None si échec)
    """
    
    # Vérifier si le fichier existe
    if not os.path.exists(image_path):
        print(f" Erreur: Le fichier n'existe pas: {image_path}")
        return None
    
    print(f"\n Traitement de l'image: {os.path.basename(image_path)}")
    
    try:
        # Charger l'image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f" Erreur: Impossible de charger l'image")
            return None
        
        # Obtenir les dimensions
        height, width, _ = img.shape
        print(f"   Dimensions: {width}x{height}")
        
        # Copie pour annotations
        result_img = img.copy()
        
        # ============================================
        # ANALYSE AVEC DEEPFACE
        # ============================================
        
        print("   🔍 Analyse avec DeepFace...")
        
        # Analyser l'image
        results = DeepFace.analyze(
            img_path=image_path, 
            actions=['emotion'], 
            enforce_detection=False,
            detector_backend='opencv'  # Plus rapide
        )
        
        # Convertir en liste si ce n'est pas déjà le cas
        if not isinstance(results, list):
            results = [results]
        
        print(f"   ✅ {len(results)} visage(s) détecté(s)")
        
        # ============================================
        # ANNOTER L'IMAGE (CORRIGÉE)
        # ============================================
        
        for idx, result in enumerate(results):
            # Émotion dominante
            emotion = result['dominant_emotion']
            
            # Toutes les émotions avec leurs scores
            emotion_scores = result['emotion']
            
            # Région du visage
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # S'assurer que les coordonnées sont valides
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            print(f"\n      Visage {idx+1}:")
            print(f"         Position: ({x}, {y}, {w}, {h})")
            print(f"         Émotion dominante: {emotion}")
            print(f"         Scores détaillés:")
            for emo, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"            {emo:10s} : {score:5.2f}%")
            
            # Couleur selon l'émotion
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            
            # Dessiner le rectangle
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
        
            # ============================================
            # CORRECTION : DÉCALAGE POUR CHAQUE VISAGE
            # ============================================
            
            # Décalage vertical différent pour chaque visage
            # Visage 1: 40px, Visage 2: 90px, Visage 3: 140px, etc.
            vertical_offset = 40 + (idx * 50)
            
            # Texte principal avec l'émotion dominante
            text = f"{emotion.upper()}: {emotion_scores[emotion]:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            
            # Calculer la position du texte
            text_y_top = y - vertical_offset
            
            # Vérifier si le texte sort de l'image en haut
            if text_y_top < 0:
                # Si trop haut, mettre en dessous du rectangle
                text_y_top = y + h + 10
                text_y_text = text_y_top + 25
            else:
                # Sinon, mettre au-dessus (position normale)
                text_y_text = text_y_top + 25
            
            # Fond pour le texte
            cv2.rectangle(
                result_img,
                (x, text_y_top),
                (x + text_size[0] + 10, text_y_top + 30),
                color,
                -1
            )
            
            # Texte
            cv2.putText(
                result_img,
                text,
                (x + 5, text_y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2
            )
        
        # ============================================
        # AFFICHER L'IMAGE
        # ============================================
        
        if display:
            print("\n   👁️ Affichage de l'image...")
            print("   💡 Touches:")
            print("      - ESC ou Q : Fermer")
            print("      - S : Sauvegarder")
            
            # Créer une fenêtre redimensionnable
            window_name = 'Emotion Recognition - DeepFace'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Ajuster la taille de la fenêtre (max 1280x720)
            display_width = min(1280, width)
            display_height = min(720, height)
            cv2.resizeWindow(window_name, display_width, display_height)
            
            # Afficher l'image
            cv2.imshow(window_name, result_img)
            
            # Attendre une touche
            saved = False
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                # ESC ou Q : Quitter
                if key == 27 or key == ord('q') or key == ord('Q'):
                    print("   👋 Fermeture de la fenêtre...")
                    break
                
                # S : Sauvegarder
                elif key == ord('s') or key == ord('S'):
                    if not saved:
                        output_path = save_result(result_img, image_path)
                        if output_path:
                            print(f"   💾 Image sauvegardée: {output_path}")
                            saved = True
                    else:
                        print("   ⚠️ Image déjà sauvegardée")
                
                # Vérifier si la fenêtre a été fermée
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            cv2.destroyAllWindows()
        
        # ============================================
        # SAUVEGARDER LE RÉSULTAT
        # ============================================
        
        if save:
            output_path = save_result(result_img, image_path)
            return output_path
        
        return None
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        print("💡 DeepFace peut télécharger des modèles la première fois.")
        print("   Cela peut prendre quelques minutes. Veuillez patienter...")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# FONCTION DE SAUVEGARDE
# ============================================

def save_result(img, original_path):
    """
    Sauvegarder l'image résultante
    
    Args:
        img: Image annotée
        original_path: Chemin de l'image originale
        
    Returns:
        str: Chemin de l'image sauvegardée
    """
    # Générer un nom de fichier unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = os.path.basename(original_path)
    name, ext = os.path.splitext(original_name)
    
    output_filename = f"{name}_deepface_{timestamp}{ext}"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    # Sauvegarder
    cv2.imwrite(output_path, img)
    
    return output_path

# ============================================
# FONCTION POUR TRAITER PLUSIEURS IMAGES
# ============================================

def process_multiple_images(image_paths):
    """
    Traiter plusieurs images successivement
    
    Args:
        image_paths (list): Liste des chemins d'images
    """
    print(f"\n📚 Traitement de {len(image_paths)} image(s)...")
    
    results = []
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n--- Image {i}/{len(image_paths)} ---")
        output_path = recognize_emotion_from_image(img_path, display=True, save=True)
        
        if output_path:
            results.append(output_path)
    
    print("\n" + "="*70)
    print(f"✅ Traitement terminé: {len(results)}/{len(image_paths)} images réussies")
    print("="*70)
    
    return results

# ============================================
# POINT D'ENTRÉE
# ============================================

if __name__ == "__main__":
    
    # Vérifier les arguments
    if len(sys.argv) > 1:
        # Si plusieurs arguments, traiter plusieurs images
        if len(sys.argv) > 2:
            image_paths = sys.argv[1:]
            print(f"📚 Mode multi-images: {len(image_paths)} image(s)")
            process_multiple_images(image_paths)
        else:
            # Une seule image
            input_image_path = sys.argv[1]
            recognize_emotion_from_image(input_image_path, display=True, save=True)
    else:
        # Image par défaut
        print(f"💡 Aucune image fournie. Utilisation de l'image par défaut:")
        print(f"   {DEFAULT_IMAGE}")
        print(f"\n💡 Usage: python {os.path.basename(__file__)} <image_path>")
        
        recognize_emotion_from_image(DEFAULT_IMAGE, display=True, save=True)
    
    print("\n✅ Programme terminé")