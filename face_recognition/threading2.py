import threading
import queue
import face_recognition
import cv2
import os
import pickle
import mediapipe as mp
import numpy as np
import time

# Global queue to hold frames for display
frame_queue = queue.Queue()

# Shared variables for controlling the execution of threads
stop_threads = False
face_recognized = [False]  # Using a list for mutable shared state

def face_web_cam():
    global stop_threads, face_recognized
   # Path to the directory containing known people's images
    KNOWN_PEOPLE_DIRECTORY = "/Users/dthomas/Documents/GitHub/Face-ASL-recognition/face_recognition_dlib/known_people"

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
                        print("No faces detected in the image!")
        # handle the situation as appropriate for your application

                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)

    # Initialize video capture via OpenCV
    video_capture = cv2.VideoCapture(0)
    print("Webcam for FR on")
    frame_count = 0

    while not stop_threads and not face_recognized[0]:
        ret, frame = video_capture.read()
        frame_count += 1

        if frame_count % 2 != 0:
            continue

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_location, face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = None
            
        if name != "Unknown":  # Assuming 'name' is the recognized face's name
            face_recognized[0] = True  # Set flag to True once face is recognized

            if min(distances) < 0.6:
                best_match_index = distances.argmin()
                name = known_face_names[best_match_index]

            # Draw box and name
            top, right, bottom, left = [coordinate * 4 for coordinate in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        frame_queue.put(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_queue.put(frame)

    video_capture.release()

def inference_classifer():
    global stop_threads, face_recognized
    model_dict = pickle.load(open('/Users/dthomas/Documents/GitHub/Face-ASL-recognition/asl-recognition-rfm/model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)
    print("Webcam for AFA on")

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
                24: 'Y', 25: 'Z'} # Complete this dictionary

    while not stop_threads and face_recognized[0]:
        ret, frame = cap.read()
        
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

            
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                # Put down here to fix multiple hands detected problems
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
                y1 = int(min(y_) * H) - 10
        
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
        
                prediction = model.predict([np.asarray(data_aux)])
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

        frame_queue.put(frame)
        cv2.waitKey(1)

        # Put processed frame into the queue instead of showing it
        frame_queue.put(frame)

    cap.release()

# Initialize threads
thread1 = threading.Thread(target=face_web_cam)
thread2 = threading.Thread(target=inference_classifer)

# Start threads
thread1.start()
thread2.start()

# Main thread for displaying the frames
try:
    while not stop_threads:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Process/display frame as needed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads = True
except KeyboardInterrupt:
    stop_threads = True

# Clean up
cv2.destroyAllWindows()
thread1.join()
thread2.join()