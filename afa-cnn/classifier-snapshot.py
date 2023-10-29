import pickle
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

model = load_model('cnn-smnist100.h5')  # Load your saved CNN model

cap = cv2.VideoCapture(0)

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

snapshot_requested = False  # Flag to indicate if a snapshot is requested
prediction_proba = 0.0  # Initialize prediction_proba

while True:
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

            data_aux = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

            # Ensure that data_aux has the correct size (28x28)
            if len(data_aux) != 28 * 28:
                # Resize or pad data_aux to match the expected input shape (28x28)
                if len(data_aux) < 28 * 28:
                    data_aux.extend([0] * (28 * 28 - len(data_aux)))
                else:
                    data_aux = data_aux[:28 * 28]

            data_aux = np.array(data_aux)
            data_aux = data_aux.reshape(1, 28, 28, 1)  # Reshape to match model input shape

            if snapshot_requested:
                prediction = model.predict([np.asarray(data_aux)])
                print('PREDICTION: ', prediction)
                predicted_character = labels_dict[
                    np.argmax(prediction)]  # Get the character with the highest probability

                prediction_proba = np.max(prediction)
                print(predicted_character, prediction_proba)
                snapshot_requested = False  # Reset the flag

            prediction_proba_pct = round(prediction_proba * 100, 2)

            cv2.putText(frame,
                        f"{predicted_character} - {prediction_proba_pct}%",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # Check if the spacebar is pressed
        snapshot_requested = True
