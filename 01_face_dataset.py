import cv2
import os

# Initialize the camera
cam = cv2.VideoCapture(0)  # Use 0 for default camera
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Load the pre-trained face detector from the OpenCV package
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# Input user ID from the terminal
face_id = input('\n Enter user ID and press <return>: ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

# Path to the dataset directory
dataset_path = 'dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Determine the highest existing image number for this user
user_files = [f for f in os.listdir(dataset_path) if f.startswith(f"User.{face_id}.") and f.endswith('.jpg')]
highest_num = 0
if user_files:
    highest_num = max([int(f.split('.')[2].split('.')[0]) for f in user_files])
start_count = highest_num + 1

# Initialize the count of individual sampled faces
count = start_count

while True:
    ret, img = cam.read()  # Read a frame from the camera
    
    # Convert the image to grayscale for the face detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces in the image
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Filename for each captured image
        filename = f"{dataset_path}/User.{face_id}.{count}.jpg"
        cv2.imwrite(filename, gray[y:y+h, x:x+w])

        # Display the resulting frame
        cv2.imshow('image', img)

    # Wait for the ESC key to exit or reach the count limit
    k = cv2.waitKey(100) & 0xff
    if k == 27:  # ESC key to break
        break
    elif count >= start_count + 200:  # Stop after capturing 200 images, adjust as needed
        break

# Cleanup the camera and close any open windows
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
