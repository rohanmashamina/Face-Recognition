import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
from face_recognition import FaceRecognitionSystem
from emotion_recognition import EmotionRecognitionSystem
import numpy as np

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Emotion Recognition System")
        self.root.geometry("800x600")
        
        self.face_system = FaceRecognitionSystem()
        self.emotion_system = EmotionRecognitionSystem()
        self.camera = None
        self.is_capturing = False
        self.current_user = ""
        self.samples_collected = 0
        self.recognition_active = False
        
        # Load eye cascade classifier
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load existing models if available
        if os.path.exists('face_model.pkl'):
            try:
                self.face_system.load_model()
                messagebox.showinfo("Info", "Loaded existing face database")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load face database: {str(e)}")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights to center content
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Camera display
        self.camera_label = ttk.Label(main_frame)
        self.camera_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Name entry frame (will be shown/hidden)
        self.name_frame = ttk.Frame(main_frame)
        self.name_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.name_frame.grid_columnconfigure(0, weight=1)
        self.name_frame.grid_columnconfigure(1, weight=1)
        
        # Configure style for larger text
        style = ttk.Style()
        style.configure('Large.TLabel', font=('Arial', 12))
        style.configure('Large.TButton', font=('Arial', 12))
        style.configure('Large.TEntry', font=('Arial', 12))
        
        ttk.Label(self.name_frame, text="Enter Name:", style='Large.TLabel').grid(row=0, column=0, pady=10, padx=10)
        self.name_entry = ttk.Entry(self.name_frame, width=30, style='Large.TEntry')
        self.name_entry.grid(row=0, column=1, pady=10, padx=10)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=20)
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        
        self.add_button = ttk.Button(buttons_frame, text="Add Face", command=self.start_face_capture, style='Large.TButton', width=15)
        self.add_button.grid(row=0, column=0, padx=20)
        
        self.recognize_button = ttk.Button(buttons_frame, text="Start Recognition", command=self.toggle_recognition, style='Large.TButton', width=15)
        self.recognize_button.grid(row=0, column=1, padx=20)
        
        self.save_button = ttk.Button(main_frame, text="Save Model", command=self.save_model, style='Large.TButton', width=20)
        self.save_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Status labels
        self.status_label = ttk.Label(main_frame, text="Ready", style='Large.TLabel')
        self.status_label.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.samples_label = ttk.Label(main_frame, text="Samples collected: 0", style='Large.TLabel')
        self.samples_label.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.debug_label = ttk.Label(main_frame, text="Debug Info: ", style='Large.TLabel')
        self.debug_label.grid(row=6, column=0, columnspan=2, pady=10)
        
        self.emotion_label = ttk.Label(main_frame, text="Emotion: ", style='Large.TLabel')
        self.emotion_label.grid(row=7, column=0, columnspan=2, pady=10)

        self.direction_label = ttk.Label(main_frame, text="Direction: ", style='Large.TLabel')
        self.direction_label.grid(row=8, column=0, columnspan=2, pady=10)

    def start_camera(self):
        """Initialize and start the camera"""
        if self.camera is None:
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    raise Exception("Could not open camera")
                # Set camera resolution
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Set camera focus
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Camera error: {str(e)}")
                return False
        return True

    def stop_camera(self):
        """Release the camera properly"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        try:
            self.camera_label.configure(image='')
        except:
            pass
        
    def get_eye_direction(self, frame, face_rect):
        x, y, w, h = face_rect
        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes with more sensitive parameters
        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.05,  # More sensitive scale factor
            minNeighbors=2,    # Even lower minimum neighbors
            minSize=(15, 15)   # Smaller minimum size
        )
        
        if len(eyes) >= 2:  # If both eyes are detected
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda x: x[0])
            left_eye, right_eye = eyes[0], eyes[1]
            
            # Calculate eye centers
            left_eye_center = (x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2)
            right_eye_center = (x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2)
            
            # Calculate the center point between eyes
            eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
            
            # Calculate the offset from face center
            face_center_x = x + w//2
            offset = eye_center_x - face_center_x
            
            # Calculate offset as percentage of face width
            offset_percentage = (offset / w) * 100
            
            # More sensitive direction detection with hysteresis
            if offset_percentage < -8:  # Eyes shifted left by more than 8%
                direction = "Looking Left"
            elif offset_percentage > 8:  # Eyes shifted right by more than 8%
                direction = "Looking Right"
            else:
                direction = "Looking Forward"
            
            # Draw eye centers and debug information
            cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)
            
            # Draw debug text
            debug_text = f"Offset: {offset_percentage:.1f}%"
            cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return direction
        else:
            # If eyes not detected, try to determine direction from face position
            face_center_x = x + w//2
            frame_center_x = frame.shape[1]//2
            offset = face_center_x - frame_center_x
            offset_percentage = (offset / frame.shape[1]) * 100
            
            if offset_percentage < -12:
                return "Looking Left"
            elif offset_percentage > 12:
                return "Looking Right"
            else:
                return "Looking Forward"

    def get_face_direction(self, frame, face_rect):
        # First try to get direction from eye tracking
        eye_direction = self.get_eye_direction(frame, face_rect)
        
        # If eye tracking fails, fall back to face position
        if eye_direction == "Looking Forward":
            x, y, w, h = face_rect
            face_center_x = x + w//2
            frame_center_x = frame.shape[1]//2
            
            # Calculate the offset from center as a percentage of frame width
            frame_width = frame.shape[1]
            offset_percentage = (face_center_x - frame_center_x) / (frame_width / 2) * 100
            
            # Determine direction based on offset percentage with adjusted thresholds
            if offset_percentage < -25:  # More than 25% to the left
                direction = "Looking Left"
            elif offset_percentage > 25:  # More than 25% to the right
                direction = "Looking Right"
            else:  # Within 25% of center
                direction = "Looking Forward"
        else:
            direction = eye_direction
            
        # Debug information
        print(f"Direction: {direction}")
            
        return direction
        
    def start_face_capture(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
            
        if not self.start_camera():
            return
            
        self.current_user = name
        self.samples_collected = 0
        self.is_capturing = True
        self.status_label.config(text=f"Capturing faces for {name}...")
        self.add_button.config(state="disabled")
        self.recognize_button.config(state="disabled")
        self.capture_faces()
        
    def capture_faces(self):
        if not self.is_capturing or self.camera is None:
            return
            
        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise Exception("Failed to capture frame")
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_system.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            # Update debug info
            self.debug_label.config(text=f"Debug Info: Faces detected: {len(faces)}")
            
            # Draw rectangles around faces and process the first detected face
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Process only the first detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add face to database
                face_img = frame[y:y+h, x:x+w]
                if self.face_system.add_face(face_img, self.current_user):
                    self.samples_collected += 1
                    self.samples_label.config(text=f"Samples collected: {self.samples_collected}")
                    
                    if self.samples_collected >= 15:
                        self.stop_face_capture()
                        messagebox.showinfo("Success", f"Added {self.current_user} to the database")
                        return
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            
            # Schedule next capture
            self.root.after(100, self.capture_faces)
            
        except Exception as e:
            print(f"Error in capture_faces: {str(e)}")
            self.stop_face_capture()
            messagebox.showerror("Error", f"Capture error: {str(e)}")
        
    def stop_face_capture(self):
        self.is_capturing = False
        self.add_button.config(state="normal")
        self.recognize_button.config(state="normal")
        self.status_label.config(text="Ready")
        self.stop_camera()
        
    def toggle_recognition(self):
        if not self.recognition_active:
            if not self.start_camera():
                return
            self.recognition_active = True
            self.recognize_button.config(text="Stop Recognition")
            self.name_frame.grid_remove()  # Hide name entry
            self.add_button.grid_remove()  # Hide add face button
            self.start_recognition()
        else:
            self.stop_recognition()
            self.recognition_active = False
            self.recognize_button.config(text="Start Recognition")
            self.name_frame.grid()  # Show name entry
            self.add_button.grid()  # Show add face button
            
    def start_recognition(self):
        if not self.start_camera():
            return
            
        self.recognition_active = True
        self.recognize_button.config(text="Stop Recognition")
        self.add_button.config(state="disabled")
        self.recognize_faces()
        
    def recognize_faces(self):
        if not self.recognition_active or self.camera is None:
            return
            
        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise Exception("Failed to capture frame")
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_system.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Get face image for recognition
                face_img = frame[y:y+h, x:x+w]
                
                # Perform face recognition
                name = self.face_system.recognize_face(face_img)
                
                # Perform emotion recognition
                try:
                    emotion, confidence, direction, has_glasses = self.emotion_system.predict_emotion(face_img)
                    if emotion is None:
                        emotion = "Unknown"
                    if confidence is None:
                        confidence = 0.0
                    if direction is None:
                        direction = "Unknown"
                except Exception as e:
                    print(f"Emotion recognition error: {str(e)}")
                    emotion = "Unknown"
                    confidence = 0.0
                    direction = "Unknown"
                    has_glasses = False
                
                # Display name, emotion, direction and glasses status
                if name:
                    cv2.putText(frame, f"Name: {name}", (x, y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if emotion:
                    cv2.putText(frame, f"Emotion: {emotion} ({confidence:.2f})", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if direction:
                    cv2.putText(frame, f"Direction: {direction}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display glasses status
                glasses_status = "Yes" if has_glasses else "No"
                cv2.putText(frame, f"Glasses: {glasses_status}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update debug info
                self.debug_label.config(text=f"Debug Info: Faces detected: {len(faces)}")
                self.emotion_label.config(text=f"Emotion: {emotion} ({confidence:.2f})")
                self.direction_label.config(text=f"Direction: {direction}")
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            
            # Schedule next recognition
            self.root.after(100, self.recognize_faces)
            
        except Exception as e:
            print(f"Error in recognize_faces: {str(e)}")
            self.stop_recognition()
        
    def stop_recognition(self):
        self.recognition_active = False
        self.stop_camera()
        self.status_label.config(text="Recognition stopped")
        self.emotion_label.config(text="Emotion: ")
        self.direction_label.config(text="Direction: ")
        
    def save_model(self):
        try:
            self.face_system.save_model()
            messagebox.showinfo("Success", "Model saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
            
    def __del__(self):
        try:
            self.stop_camera()
        except:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop() 