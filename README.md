# Face Recognition System

## Overview
This Python script utilizes computer vision techniques and machine learning models to perform face recognition. It detects faces in a video stream or images using MediaPipe's face detection model, extracts facial embeddings using the FaceNet model,  and compares them with known embeddings to recognize individuals.

## Features
- Real-time face detection and recognition from webcam feed.
- Recognition of known individuals based on facial embeddings.
- Adjustable confidence threshold for recognition accuracy.
- Scalable for recognition of multiple individuals.

## Requirements
- Python 3.x
- OpenCV 
- NumPy
- Mediapipe
- Keras Facenet
- SciPy
- TensorFlow

## Installation
1. Install Python 3.x from [python.org](https://www.python.org/).
2. Install required libraries using pip:
   ```bash
   pip install opencv-python mediapipe numpy keras-facenet scipy tensorflow

## Clone or download the repository:
3. git clone <repository-url>


## Usage
- Add images of known individuals to the ps folder, organized in subfolders named after each person.
- Run the script:
- python face_recognition.py

## Parameters
- Confidence Threshold: Adjust the confidence_threshold variable in the script to control the recognition accuracy.

## Reference
- FaceNet model implementation: Keras Facenet