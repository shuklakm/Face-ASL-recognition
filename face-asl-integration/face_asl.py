#face
import face_recognition
import os
import numpy as np
from face_recognition import face_distance

#asl 
import pickle
import cv2
import mediapipe as mp
import subprocess


# --------------------- Initialize Face Recognition ---------------------

# Path to the directory containing known people's images
KNOWN_PEOPLE_DIRECTORY = "/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/known_people"

# Initialize known faces and encodings
known_face_encodings = []
known_face_names = []
frame_count = 0

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
                




# --------------------- Initialize ASL Recognition ---------------------

model_dict = pickle.load(open('/Users/kajalshukla/Desktop/Face-ASL-recognition/asl-recognition-rfm/model_both.p', 'rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C',
               3: 'D', 4: 'E', 5: 'F',
               6: 'G', 7: 'H', 8: 'I',
               9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 
               15: 'P', 16: 'Q', 17: 'R', 
               18: 'S', 19: 'T', 20: 'U', 
               21: 'V', 22: 'W', 23: 'X', 
               24: 'Y', 25: 'Z'}

# Variables for gesture confirmation
confirmation_threshold = 15  # Number of frames to confirm the gesture
gesture_counter = 0
current_gesture = None

video_capture = cv2.VideoCapture(0)


# --------------------- Merge FACE-ASL Recognition ---------------------

while True:
    ret, frame = video_capture.read()
    frame_count +=1
    
    if frame_count % 2 !=0:
        continue

    # Initialize flag for known face detection
    known_face_detected = False

    # Process for Face Recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        distances = face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.5:
            name = known_face_names[best_match_index]
            known_face_detected = True  # Set flag to True for known face
        else:
            name = "Unknown"

        top, right, bottom, left = [coordinate * 4 for coordinate in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    
    

    # Process for ASL Recognition
    if known_face_detected:
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                    )
                
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                    
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    
                    x_.append(x)
                    y_.append(y)
                    
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(max(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction  = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                prediction_proba = model.predict_proba([np.asarray(data_aux)])
                print(predicted_character, prediction_proba)

                prediction_proba_pct = round(max(prediction_proba[0]) * 100,2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame,
                            f"{predicted_character} - {prediction_proba_pct}%",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
                
                # TTS
                if predicted_character == current_gesture:
                    gesture_counter += 1
                    if gesture_counter == confirmation_threshold:
                        subprocess.run(['say', predicted_character], check=True)
                        gesture_counter = 0
                else:
                    current_gesture = predicted_character
                    gesture_counter = 0


    # Display the processed frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

