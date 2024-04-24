import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the face recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load the pre-trained face detection model
faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

# Font style for text on the video
font = cv2.FONT_HERSHEY_SIMPLEX

# Names corresponding to the IDs, adjust as needed
names = ['None', 'David', 'Yazeed', 'Mum']

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

def init():
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 1.1 * max(id_count))
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
    ax2.set_ylim(0, 1.1 * max(id_count))
    return []

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=100)

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

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Exit if ESC is pressed
        break

    plt.pause(0.01)

print("\n [INFO] Exiting Program ...")
cam
