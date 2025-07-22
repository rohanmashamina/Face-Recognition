# Face & Emotion Recognition System

This is a Python-based project for **real-time face and emotion recognition** using **deep learning (PyTorch)**, **OpenCV**, and a **Tkinter GUI**. It detects faces through your webcam or video stream, predicts emotional states, estimates face direction, and identifies the presence of glasses.

---

## 🧠 Features

* 🎥 Real-time emotion recognition via webcam
* 🙂 Face detection with emotion classification (7 emotions)
* 🧭 Face direction estimation (left, right, center)
* 🕶️ Glasses detection
* 🖥️ GUI interface using Tkinter
* 🧪 Model training and evaluation from scratch (optional)
* ⚙️ Uses CNN for emotion classification (PyTorch)
* 👁️ Face recognition using OpenCV Haar cascades and LBPH

---

## 🗂️ Project Structure

```
ml project/
│
├── app.py                      # GUI application
├── emotion_recognition.py      # Main emotion recognition logic
├── face_recognition.py         # Face recognition logic
├── train.py                    # Model training script
├── requirements.txt            # Python dependencies
├── emotion_model.pth           # Trained emotion model (PyTorch)
├── face_recognizer_model.yml   # Trained face recognizer (OpenCV)
├── archive/                    # Dataset folders
│   └── train/                  # Training images (by emotion)
│   └── test/                   # Test images (by emotion)
├── saved_faces/                # Saved face images (by person)
└── README.md                   # Project documentation
```

## 🔧 Installation

1. **Clone the repository**

   git clone https://github.com/rohanmashamina/Face-Recognition.git
   cd "ml project"
  

2. **Install dependencies**

   
   pip install -r requirements.txt


3. **Prepare the dataset**

   * Place your **training** and **test** images inside the `archive/train/` and `archive/test/` folders respectively.
   * Organize them into folders named after emotion classes like:
     angry/
     happy/
     sad/
     surprise/
     fear/
     disgust/
     neutral/


4. **Train the model** (if needed)

   * If the file `emotion_model.pth` does not exist, the model will train automatically.
   * Or run manually:

     python train.py
     

## 🚀 Usage

### Run Emotion Recognition

python emotion_recognition.py


* Starts real-time emotion detection via webcam.
* Detects face direction and glasses.
* Press `q` to exit the webcam window.

### Run GUI Application

python app.py

* A GUI window will launch.
* Provides access to real-time face and emotion recognition tools via a user-friendly interface.

## 🧠 Model Details

* **Architecture**: Custom CNN defined in `emotion_recognition.py` (`EmotionCNN` class)
* **Input**: 48x48 grayscale images
* **Framework**: PyTorch
* **Emotion Classes**:

  * angry
  * disgust
  * fear
  * happy
  * neutral
  * sad
  * surprise

## 📦 Requirements

* Python 3.8+
* OpenCV
* PyTorch
* scikit-learn
* numpy
* tkinter

Refer to [`requirements.txt`](./requirements.txt) for full package list.

## 📌 Notes

* If `emotion_model.pth` is missing, the model will train automatically on startup.
* Face detection uses **OpenCV Haar cascades**.
* Face recognition uses **LBPH recognizer** from OpenCV.
* The GUI is implemented in `app.py` (`FaceRecognitionApp` class).
