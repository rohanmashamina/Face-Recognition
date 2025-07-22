# Facial Emotion Recognition System

This project implements a real-time facial emotion recognition system using a Multi-Layer Perceptron (MLP) neural network. The system can detect and classify four basic emotions: angry, happy, neutral, and sad.

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`):
  - opencv-python
  - numpy
  - tensorflow
  - pandas
  - scikit-learn

## Dataset

The system uses the FER-2013 dataset. You need to download the dataset and place the `fer2013.csv` file in the project directory.

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the FER-2013 dataset and place it in the project directory

## Usage

1. Train the model:
   ```bash
   python emotion_recognition.py
   ```
   This will:
   - Load and preprocess the FER-2013 dataset
   - Train the MLP model
   - Start real-time emotion prediction using your webcam

2. During real-time prediction:
   - The system will display your face with the predicted emotion
   - Press 'q' to quit the application

## Model Architecture

The MLP model consists of:
- Input layer: 2304 nodes (48x48 flattened image)
- Hidden layer 1: 512 nodes with ReLU activation
- Dropout layer: 0.2
- Hidden layer 2: 256 nodes with ReLU activation
- Dropout layer: 0.2
- Output layer: 4 nodes with softmax activation (one for each emotion)

## Performance

The model achieves reasonable accuracy on the validation set. For better performance, you can:
- Increase the number of training epochs
- Adjust the model architecture
- Use data augmentation
- Fine-tune hyperparameters

## License

This project is open source and available under the MIT License. #

