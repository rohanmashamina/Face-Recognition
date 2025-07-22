import cv2
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_faces = []
        self.known_names = []
        self.label_encoder = LabelEncoder()
        self.svm_classifier = SVC(kernel='linear', probability=True)
        self.is_trained = False
        self.saved_faces_dir = 'saved_faces'
        self.face_count = {}  # Track number of samples per person
        self.target_size = (150, 150)  # Increased size for better features
        
        # Create saved_faces directory if it doesn't exist
        if not os.path.exists(self.saved_faces_dir):
            os.makedirs(self.saved_faces_dir)
            
        # Load any existing saved faces
        self.load_saved_faces()
        
        self.load_model()

    def load_saved_faces(self):
        """Load faces from saved_faces directory"""
        for person_dir in os.listdir(self.saved_faces_dir):
            person_path = os.path.join(self.saved_faces_dir, person_dir)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(person_path, img_file)
                        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if face_img is not None:
                            # Resize to consistent size
                            face_img = cv2.resize(face_img, (100, 100))
                            self.known_faces.append(face_img)
                            self.known_names.append(person_dir)
        
        if self.known_faces:
            self.train_model()
            
    def add_face(self, face_img, name):
        """Add a new face to the database"""
        try:
            # Convert to grayscale if needed
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
            # Resize face to consistent size
            face_img = cv2.resize(face_img, (100, 100))
            
            # Save the face image
            person_dir = os.path.join(self.saved_faces_dir, name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                
            # Save the image
            img_path = os.path.join(person_dir, f"{len(os.listdir(person_dir))}.jpg")
            cv2.imwrite(img_path, face_img)
            
            # Add to training data
            self.known_faces.append(face_img)
            self.known_names.append(name)
            
            # Retrain the model
            self.train_model()
            
            return True
            
        except Exception as e:
            print(f"Error adding face: {str(e)}")
            return False
            
    def train_model(self):
        """Train the face recognition model"""
        if not self.known_faces:
            return
            
        try:
            # Convert lists to numpy arrays
            faces = np.array(self.known_faces)
            labels = self.label_encoder.fit_transform(self.known_names)
            
            # Train the model
            self.face_recognizer.train(faces, labels)
            self.is_trained = True
            
            # Save the model
            self.save_model()
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            self.is_trained = False
            
    def recognize_face(self, face_img):
        """Recognize a face and return the name and confidence"""
        if not self.is_trained:
            return "Unknown", 0.0
            
        try:
            # Convert to grayscale if needed
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
            # Resize face to consistent size
            face_img = cv2.resize(face_img, (100, 100))
            
            # Apply histogram equalization for better contrast
            face_img = cv2.equalizeHist(face_img)
            
            # Predict using the trained model
            label, confidence = self.face_recognizer.predict(face_img)
            
            # Convert confidence to a 0-1 scale (lower is better in LBPH)
            confidence = 1.0 - min(confidence / 100.0, 1.0)
            
            # Print debug information
            print(f"Recognition - Label: {label}, Confidence: {confidence}")
            
            if confidence > 0.3:  # Lowered threshold for better recognition
                try:
                    name = self.label_encoder.inverse_transform([label])[0]
                    print(f"Recognized as: {name}")
                    return name, confidence
                except Exception as e:
                    print(f"Error in label transformation: {str(e)}")
                    return "Unknown", confidence
            else:
                print("Recognition confidence too low")
                return "Unknown", confidence
                
        except Exception as e:
            print(f"Error recognizing face: {str(e)}")
            return "Unknown", 0.0
            
    def save_model(self):
        """Save the trained model"""
        try:
            # Save the model file
            self.face_recognizer.save('face_recognizer_model.yml')
            
            # Save the label encoder and other data
            with open('face_recognition_data.pkl', 'wb') as f:
                pickle.dump({
                    'label_encoder': self.label_encoder,
                    'known_names': self.known_names
                }, f)
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            
    def load_model(self):
        """Load a trained model"""
        try:
            if os.path.exists('face_recognizer_model.yml') and os.path.exists('face_recognition_data.pkl'):
                # Load the model file
                self.face_recognizer.read('face_recognizer_model.yml')
                
                # Load the label encoder and other data
                with open('face_recognition_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.label_encoder = data['label_encoder']
                    self.known_names = data['known_names']
                    
                self.is_trained = True
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_trained = False

    def preprocess_face(self, face_img):
        try:
            if face_img is None or face_img.size == 0:
                print("Invalid input image")
                return None
                
            # Resize to consistent size
            face_img = cv2.resize(face_img, self.target_size)
            
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            # Apply histogram equalization
            gray = cv2.equalizeHist(gray)
            
            # Apply adaptive thresholding
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Normalize pixel values
            gray = gray.astype(np.float32) / 255.0
            
            # Flatten the image to a 1D array
            features = gray.reshape(1, -1)
            
            # Verify the feature dimensions
            expected_features = self.target_size[0] * self.target_size[1]
            if features.shape[1] != expected_features:
                print(f"Unexpected feature dimension: got {features.shape[1]}, expected {expected_features}")
                return None
                
            return features
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None

    def update_face_count(self, name):
        if name not in self.face_count:
            self.face_count[name] = 0
        self.face_count[name] += 1

    def save_model_data(self):
        try:
            model_data = {
                'X': self.X,
                'y': self.y,
                'face_count': self.face_count
            }
            with open('face_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model_data(self):
        try:
            if os.path.exists('face_model.pkl'):
                with open('face_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                    self.X = model_data['X']
                    self.y = model_data['y']
                    self.face_count = model_data.get('face_count', {})
                    self.face_recognizer.fit(self.X, self.y)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.X = None
            self.y = None
            self.face_count = {} 