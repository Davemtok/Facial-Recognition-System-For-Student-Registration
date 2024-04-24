import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

def getImagesAndLabels(path, detector):
    # Get list of image paths in the dataset directory
    imagePaths = list(Path(path).glob('*'))
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            # Convert image to grayscale
            PIL_img = Image.open(str(imagePath)).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract ID from filename
            id = int(imagePath.stem.split(".")[1])

            # Detect faces in the image
            faces = detector.detectMultiScale(img_numpy, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Crop to the face area
                face_img = img_numpy[y:y+h, x:x+w]

                # Draw a rectangle around the face
                cv2.rectangle(img_numpy, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue color rectangle with thickness 2

                # Add face samples and corresponding IDs to lists
                faceSamples.append(face_img)
                ids.append(id)

                # Display the image with the face highlighted
                cv2.imshow("Training on image...", cv2.resize(img_numpy, (400, 400)))
                cv2.waitKey(100)  # Waits for 100 ms for the user to see the image, press any key to continue
        except Exception as e:
            # Handle exceptions (e.g., corrupted file)
            print(f"Error processing {imagePath}: {e}")

    # Destroy the created window once all images are processed
    cv2.destroyAllWindows()

    return faceSamples, ids

def main():
    # Path to face image database
    path = 'dataset'

    # Create a face recognizer and a face detector
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

    # Inform the user that training is about to start
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

    # Get the faces and IDs from the dataset
    faces, ids = getImagesAndLabels(path, detector)

    # Train the recognizer with the faces and IDs
    recognizer.train(faces, np.array(ids))

    # Save the trained model
    recognizer.write('trainer/trainer.yml')

    # Print the number of faces trained and exit
    print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")

# Check if this script is being run directly
if __name__ == "__main__":
    main()