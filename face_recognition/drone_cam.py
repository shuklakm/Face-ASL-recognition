# use face_asl_recognition environment

from djitellopy import Tello
import cv2
import numpy as np
import face_recognition
from face_recognition import face_distance
import os

known_people_folder = "/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/known_people"


def load_known_faces(known_people_folder):
    known_names = []
    known_face_encodings = []
    for name in os.listdir(known_people_folder):
        subdir = os.path.join(known_people_folder, name)
        if os.path.isdir(subdir):
            for filename in os.listdir(subdir):
                if filename.endswith(".jpg"):
                    file_path = os.path.join(subdir, filename)
                    image = face_recognition.load_image_file(file_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                    else:
                        print("No faces detected in the image")
                        continue

                    known_face_encodings.append(face_encoding)
                    known_names.append(name)
    return known_names, known_face_encodings

# Initialize and connect to the Tello drone
tello = Tello("192.168.0.167")
tello.connect()
tello.streamon()  # Start video streaming

known_names, known_face_encodings = load_known_faces(known_people_folder)

# Use the frame from the Tello drone
def recognize_faces_in_tello_video(known_names, known_face_encodings, tolerance=0.5):
    frame_count = 0
    scale_factor = 0.25  # scale factor for resizing

    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1

        # Process every 2nd frame
        if frame_count % 2 != 0:
            continue

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        face_locations = face_recognition.face_locations(small_frame, model="cnn")
        unknown_face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_face_encodings):
            distances = face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(distances)

            name = "Unknown"
            if distances[best_match_index] < tolerance:
                name = known_names[best_match_index]

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= int(1/scale_factor)
            right *= int(1/scale_factor)
            bottom *= int(1/scale_factor)
            left *= int(1/scale_factor)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        # Display the resulting frame
        cv2.imshow('Tello Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()


# known_names, known_face_encodings = load_known_faces(known_people_folder)
recognize_faces_in_tello_video(known_names, known_face_encodings)
