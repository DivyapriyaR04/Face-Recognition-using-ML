# Face Recognition and Attendance System Using Machine Learning
This project implements a face recognition system using machine learning to capture images, train a recognition model, and mark attendance. It leverages Python, OpenCV, and dlib for various stages of face detection, feature extraction, and recognition.

# Features
### Image Capture: 
Captures multiple images of a person using a webcam.
### Face Detection: 
Detects faces in images using dlib's face detector.
### Feature Extraction: 
Extracts face embeddings using a pre-trained dlib model.
### Model Training:
Trains a Support Vector Machine (SVM) classifier on the extracted face embeddings.
### Face Recognition:
Recognizes faces in real-time video streams.
### Attendance Marking: 
Marks attendance in a CSV file based on recognized faces.

# Install Dependencies:
```
pip install -r requirements.txt
```

# Download Pre-trained Models:
```
shape_predictor_68_face_landmarks.dat
dlib_face_recognition_resnet_model_v1.dat
```
on github and store these models inside 'models' directory
# Project Structure:
```
Face-Recognition-using-ML/
│
├── datasets.py                      # Script to capture images
├── face_recong.py                   # Script to train the recognition model and for real-time recognition and attendance marking
│
├── models/                          # Directory to store pre-trained models
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── dataset/                         # Directory to store captured images
│
├── attendance.csv                   # CSV file to store attendance records
├── requirements.txt                 # Project dependencies
└── README.md                        # Project description and usage guide
```


