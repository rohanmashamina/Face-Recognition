import os
import cv2
import numpy as np
from face_recognition import FaceRecognitionSystem
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(dataset_path):
    """Load face images and labels from a dataset directory"""
    faces = []
    labels = []
    
    print(f"Loading dataset from: {dataset_path}")
    print("Found the following persons:")
    
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            print(f"\nPerson: {person_name}")
            image_count = 0
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    img = cv2.imread(image_path)
                    if img is not None:
                        faces.append(img)
                        labels.append(person_name)
                        image_count += 1
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
            print(f"Loaded {image_count} images")
    
    print(f"\nTotal faces loaded: {len(faces)}")
    print(f"Total unique persons: {len(set(labels))}")
    
    return np.array(faces), np.array(labels)

def train_model(dataset_path, model_save_path):
    """Train the face recognition model"""
    print("Loading dataset...")
    faces, labels = load_dataset(dataset_path)
    
    # Initialize face recognition system
    face_system = FaceRecognitionSystem()
    
    print("Processing faces...")
    processed_faces = []
    processed_labels = []
    
    for face, label in zip(faces, labels):
        face_boxes = face_system.detect_faces(face)
        if face_boxes:
            x, y, w, h = face_boxes[0]
            face_img = face[y:y+h, x:x+w]
            processed_faces.append(face_img)
            processed_labels.append(label)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed_faces, processed_labels, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    for face, label in zip(X_train, y_train):
        face_system.add_face(face, label)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = []
    for face in X_test:
        label, _ = face_system.recognize_face(face)
        y_pred.append(label)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    
    # Save model
    print(f"\nSaving model to {model_save_path}...")
    face_system.save_model(model_save_path)
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train face recognition model')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--model', type=str, default='face_model.pkl',
                      help='Path to save the trained model')
    
    args = parser.parse_args()
    
    train_model(args.dataset, args.model) 