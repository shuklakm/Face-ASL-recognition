import face_recognition
import cv2
import os
import numpy as np
from face_recognition import face_distance

# Path to the directory containing known people's images
KNOWN_PEOPLE_DIRECTORY = "/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/known_people"

# Initialize known faces and encodings
known_face_encodings = []
known_face_names = []

# Load images and encode them
for name in os.listdir(KNOWN_PEOPLE_DIRECTORY):
    person_dir = os.path.join(KNOWN_PEOPLE_DIRECTORY, name)
    if os.path.isdir(person_dir):
        for filename in os.listdir(person_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                     face_encoding = face_encodings[0]
                else:
                    print(f"No faces detected in the image {person_dir} {filename}!")
    # handle the situation as appropriate for your application

                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

# Initialize video capture via OpenCV
video_capture = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = video_capture.read()
    frame_count += 1

    # Process every 2nd frame
    if frame_count % 2 != 0:
        continue

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = face_recognition.face_locations(small_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        distances = face_distance(known_face_encodings, face_encoding)
        
        best_match_index = np.argmin(distances)        
        if distances[best_match_index] < 0.5:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"

        # Draw box and name
        top, right, bottom, left = [coordinate * 4 for coordinate in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()