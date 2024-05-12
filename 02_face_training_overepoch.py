import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt

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

                # Add face samples and corresponding IDs to lists
                faceSamples.append(face_img)
                ids.append(id)
        except Exception as e:
            # Handle exceptions (e.g., corrupted file)
            print(f"Error processing {imagePath}: {e}")

    return faceSamples, ids

def trainModel(path):
    # Create a face recognizer and a face detector
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

    # Lists to store accuracy and loss
    accuracies = []
    losses = []
    
    # Variables to track best accuracy and corresponding epoch
    best_accuracy = 0
    best_epoch = 0

    # Inform the user that training is about to start
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

    epochs_without_improvement = 0
    max_epochs_without_improvement = 10  # Change this value as needed

    epoch = 1
    while epochs_without_improvement < max_epochs_without_improvement:
        # Get the faces and IDs from the dataset
        faces, ids = getImagesAndLabels(path, detector)

        # Train the recognizer with the faces and IDs
        recognizer.train(faces, np.array(ids))

        # Calculate accuracy
        total_faces = len(faces)
        correct_predictions = 0
        for face, id in zip(faces, ids):
            label, confidence = recognizer.predict(face)
            if label == id:
                correct_predictions += 1
        accuracy = (correct_predictions / total_faces) * 100
        
        # Save accuracy and loss
        accuracies.append(accuracy)
        losses.append(100 - accuracy)

        # Print current epoch and accuracy
        print(f"Epoch {epoch}: Accuracy = {accuracy:.2f}%")

        # Check if current accuracy is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            recognizer.write('trainer/best_trained_model.yml')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch += 1

    print(f"\nBest Accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")

    # Plot accuracy and loss
    plt.plot(range(1, epoch), accuracies, label='Accuracy')
    plt.plot(range(1, epoch), losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.title('Accuracy and Loss over Epochs')
    plt.legend()
    plt.show()

# Check if this script is being run directly
if __name__ == "__main__":
    # Path to face image database
    dataset_path = 'dataset'

    # Create directory to save the best trained model if not exists
    os.makedirs('trainer', exist_ok=True)

    # Train the model
    trainModel(dataset_path)
