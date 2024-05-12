import cv2
import os

# Load the face detection model
faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

# Specify the path to the dataset folder
dataset_folder_path = 'test_images'
if not os.path.exists(dataset_folder_path):
    os.makedirs(dataset_folder_path)

# Track IDs and sequence numbers
face_id_tracker = {}

# Process each image in the folder
for filename in os.listdir(dataset_folder_path):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Check for image files
        img_path = os.path.join(dataset_folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue  # skip files that aren't images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Extract the face ID from the filename if possible
        file_parts = filename.split('.')
        if len(file_parts) >= 3 and file_parts[0] == "User":
            face_id = int(file_parts[1])
        else:
            continue  # skip files that do not follow the naming convention

        # Initialize or update face sequence for this ID
        if face_id not in face_id_tracker:
            max_seq = 0
            # Check existing files to determine the next sequence number
            for f in os.listdir(dataset_folder_path):
                if f.startswith(f"User.{face_id}.") and f.endswith('.jpg'):
                    try:
                        seq_num = int(f.split('.')[2].split('.')[0])
                        if seq_num > max_seq:
                            max_seq = seq_num
                    except ValueError:
                        continue
            face_id_tracker[face_id] = max_seq

        # Save each detected face as a new image
        for i, (x, y, w, h) in enumerate(faces):
            face_id_tracker[face_id] += 1
            face_img = gray[y:y+h, x:x+w]  # Crop the grayscale image
            face_filename = f"User.{face_id}.{face_id_tracker[face_id]}.jpg"
            face_path = os.path.join(dataset_folder_path, face_filename)
            cv2.imwrite(face_path, face_img)  # Save the cropped grayscale face

print("\n [INFO] Face extraction and saving complete.")
