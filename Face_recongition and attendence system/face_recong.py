import os
import cv2
import dlib
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv
from datetime import datetime

# Paths
dataset_path = 'dataset/'
shape_predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'models/dlib_face_recognition_resnet_model_v1.dat'

# Initialize dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()

# Check if the shape predictor file exists
if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Shape predictor file not found: {shape_predictor_path}")

sp = dlib.shape_predictor(shape_predictor_path)

# Check if the face recognition model file exists
if not os.path.exists(face_rec_model_path):
    raise FileNotFoundError(f"Face recognition model file not found: {face_rec_model_path}")

facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Function to get face embeddings
def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    for face in faces:
        shape = sp(gray, face)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)

# Load dataset and extract features
labels = []
features = []
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        embeddings = get_face_embeddings(img_path)
        if embeddings is not None:
            features.append(embeddings)
            labels.append(person)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Check the number of unique labels
unique_labels = np.unique(labels)
print("Unique labels:", unique_labels)

if len(unique_labels) > 1:
    # Encode the labels
    le = preprocessing.LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    # Train a SVM classifier
    clf = SVC(C=1.0, kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Function to mark attendance
    def mark_attendance(name):
        with open('attendance.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    # Start video capture
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            shape = sp(gray, face)
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            face_descriptor = np.array(face_descriptor).reshape(1, -1)
            prediction = clf.predict(face_descriptor)
            name = le.inverse_transform(prediction)[0]
            mark_attendance(name)
            # Draw rectangle around the face
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("The dataset has only one class. The number of classes must be greater than one to train the classifier.")
