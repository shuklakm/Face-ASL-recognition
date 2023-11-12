import face_recognition
import cv2
import os
from djitellopy import Tello

# Path to the directory containing known people's images
KNOWN_PEOPLE_DIRECTORY = "/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/known_people"

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
                    print("No faces detected in the image!")
                    continue

                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

tello = Tello('192.168.0.167')
tello.connect()
tello.streamon()

frame_count = 0

while True:
    frame = tello.get_frame_read().frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_count += 1

    if frame_count % 2 != 0:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = face_recognition.face_locations(small_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = None
        name = "Unknown"
        if min(distances) < 0.5:
            best_match_index = distances.argmin()
            name = known_face_names[best_match_index]

        top, right, bottom, left = [coordinate * 4 for coordinate in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        

    cv2.imshow('Tello Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
