import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Load the trained model if it exists, else create an empty model
if os.path.exists('trainer/best_trained_model.yml'):
    recognizer.read('trainer/best_trained_model.yml')
else:
    recognizer.train([], np.array([]))

# Load the pre-trained face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Font style for text on the video
font = cv2.FONT_HERSHEY_SIMPLEX

# Names corresponding to the IDs, adjust as needed
names = ['None', 'David', 'Ninja']

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define minimum window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Prepare a figure to host the visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
id_count = np.zeros(len(names))

# Data storage for confidence graph
confidences = []

# Excel setup
excel_file = 'Face_Records.xlsx'
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=['Name/Date', datetime.now().strftime('%Y-%m-%d')])
    df.to_excel(excel_file, index=False)
else:
    df = pd.read_excel(excel_file)

# Set to keep track of recognized faces for the day
recognized_faces_set = set(df['Name/Date'])

def init():
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, len(names))
    return []

def update(frame):
    if len(confidences) > 0:
        avg_conf = sum(confidences) / len(confidences)
        ax1.clear()
        ax1.plot(range(len(confidences)), confidences, label='Confidence')
        ax1.legend()
        ax1.set_ylim(0, 100)

    ax2.clear()
    ax2.bar(names, id_count, color='blue')
    ax2.set_ylim(0, len(names))
    return []

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=100)

# Variable to keep track of training data
training_data = []

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 100:
            name_id = names[id]
            confidence_text = "  {0}%".format(round(100 - confidence))
            # Append the face region and corresponding label for training
            training_data.append((gray[y:y+h, x:x+w], id))
        else:
            name_id = "unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(name_id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        # Update the ID counts and confidences
        if name_id in names:
            id_index = names.index(name_id)
            id_count[id_index] += 1
            confidences.append(100 - confidence)

            # Log the attendance if the face has not been recorded today
            if name_id not in recognized_faces_set:
                recognized_faces_set.add(name_id)
                date_str = datetime.now().strftime('%Y-%m-%d')
                new_record = pd.DataFrame({'Name/Date': [name_id], date_str: ['Present']})
                df = pd.concat([df, new_record], ignore_index=True)
                df.to_excel(excel_file, index=False)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Exit if ESC is pressed
        break

    # Train the recognizer with the new data every epoch
    if len(training_data) > 0:
        faces = []
        labels = []
        for face, label in training_data:
            faces.append(face)
            labels.append(label)
        recognizer.update(faces, np.array(labels))
        training_data = []

    plt.pause(0.01)

print("\n [INFO] Exiting Program ...")
cam.release()
cv2.destroyAllWindows()

