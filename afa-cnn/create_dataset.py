import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data/train1'

data = []
labels = []

dir_list = os.listdir(DATA_DIR)
dir_list = sorted(dir_list)

for dir_ in dir_list:
    if dir_ != ".DS_Store":
        count = 0
        img_path_list = os.listdir(os.path.join(DATA_DIR, dir_))
        img_path_list = sorted(img_path_list)
        for img_path in img_path_list:
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))

            if img is None:
                print(img_path)
            if img is not None:
                count += 1
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    if len(results.multi_hand_landmarks) > 1:
                        print(f"{img_path} detected {str(len(results.multi_hand_landmarks))}")
                    hand_landmarks = results.multi_hand_landmarks[0]
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

                    data.append(data_aux)
                    labels.append(dir_)

        print("Done: ", dir_)
        print(count)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()