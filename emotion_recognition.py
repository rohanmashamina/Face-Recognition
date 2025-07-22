import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split

class EmotionDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return image, label

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 7)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Initialize the model
        self.model = EmotionCNN().to(self.device)
        
        # Try to load pre-trained model
        if os.path.exists('emotion_model.pth'):
            try:
                self.model.load_state_dict(torch.load('emotion_model.pth', map_location=self.device))
                self.model.eval()
                print("Loaded pre-trained emotion model")
            except Exception as e:
                print(f"Error loading pre-trained model: {str(e)}")
                print("Training new model...")
                self.train_new_model()
        else:
            print("No pre-trained model found. Training new model...")
            self.train_new_model()
            
    def train_new_model(self):
        """Train a new emotion recognition model"""
        try:
            # Load and preprocess data
            train_loader, val_loader = load_and_preprocess_data('archive/train')
            
            # Train the model
            self.train_model(train_loader, val_loader)
            
            # Save the trained model
            torch.save(self.model.state_dict(), 'emotion_model.pth')
            print("Saved trained model")
        except Exception as e:
            print(f"Error training new model: {str(e)}")
            print("Using untrained model - emotion recognition may not work properly")
            self.model.eval()
        
    def get_face_direction(self, gray, face_rect):
        """Determine the direction of the face using eye detection"""
        x, y, w, h = face_rect
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda x: x[0])
            
            # Calculate eye positions relative to face width
            left_eye_x = eyes[0][0] / w
            right_eye_x = eyes[1][0] / w
            
            # Calculate average eye y-position relative to face height
            avg_eye_y = (eyes[0][1] + eyes[1][1]) / (2 * h)
            
            # Determine horizontal direction
            if left_eye_x < 0.3:
                return "right"
            elif right_eye_x > 0.7:
                return "left"
            
            # Determine vertical direction
            if avg_eye_y < 0.3:
                return "up"
            elif avg_eye_y > 0.45:
                return "down"
                
            return "front"
        else:
            # If eyes not detected clearly, try to determine direction from face position
            face_center_x = x + w//2
            face_center_y = y + h//2
            
            # Check vertical position first
            if face_center_y < h * 0.3:
                return "up"
            elif face_center_y > h * 0.7:
                return "down"
            
            # Then check horizontal position
            if face_center_x < w * 0.3:
                return "right"
            elif face_center_x > w * 0.7:
                return "left"
            
            return "front"  # Default if no clear direction detected

    def predict_emotion(self, image):
        """Predict emotion from a single image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return "neutral", 0.0, "front", False
            
        # Get the first face
        (x, y, w, h) = faces[0]
        
        # Get face direction
        face_direction = self.get_face_direction(gray, (x, y, w, h))
        
        # Check for glasses
        roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        has_glasses = len(eyes) < 2  # If less than 2 eyes detected, likely wearing glasses
        
        # Crop and resize face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        
        # Normalize pixel values
        face = face.astype('float32') / 255.0
        
        # Reshape for CNN input (add channel dimension)
        face = face.reshape(1, 48, 48)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(face).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
            
            # If confidence is too low, return neutral
            if confidence < 0.2:  # Lowered threshold from 0.3 to 0.2
                print(f"Emotion: neutral (confidence: {confidence:.2f}), Direction: {face_direction}, Glasses: {'Yes' if has_glasses else 'No'}")
                return "neutral", confidence, face_direction, has_glasses
                
            emotion = self.emotion_labels[predicted.item()]
            print(f"Emotion: {emotion} (confidence: {confidence:.2f}), Direction: {face_direction}, Glasses: {'Yes' if has_glasses else 'No'}")
            return emotion, confidence, face_direction, has_glasses
        
    def run_realtime_prediction(self):
        """Run real-time emotion prediction using webcam"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Predict emotion, face direction, and glasses status
            emotion, confidence, direction, has_glasses = self.predict_emotion(frame)
            
            if emotion is not None:
                # Draw emotion text on frame
                text = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw face direction
                cv2.putText(frame, f"Direction: {direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw glasses status
                cv2.putText(frame, f"Glasses: {'Yes' if has_glasses else 'No'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # Display the frame
            cv2.imshow('Emotion Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def load_and_preprocess_data(data_dir):
    """Load and preprocess the image dataset"""
    print(f"Loading data from directory: {data_dir}")
    X = []
    y = []
    
    # Map emotion folders to labels
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    
    # Process each emotion folder
    for emotion_folder, label in emotion_map.items():
        folder_path = os.path.join(data_dir, emotion_folder)
        print(f"Processing folder: {folder_path}")
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            continue
            
        file_count = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Read image
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                    
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize to 48x48
                gray = cv2.resize(gray, (48, 48))
                
                # Normalize
                gray = gray.astype('float32') / 255.0
                
                # Reshape for CNN
                gray = gray.reshape(48, 48)
                
                X.append(gray)
                y.append(label)
                file_count += 1
                
        print(f"Processed {file_count} images from {emotion_folder}")
    
    X = np.array(X)
    y = np.array(y)
    print(f"Total images loaded: {len(X)}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Starting emotion recognition system...")
    # Initialize the system
    emotion_system = EmotionRecognitionSystem()
    print("System initialized")
    
    # Import and run the GUI application
    import tkinter as tk
    from app import FaceRecognitionApp
    
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop() 