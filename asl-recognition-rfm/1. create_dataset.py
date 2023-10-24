#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 22:15:31 2023

@author: trangdang
"""

# Set up environment 
import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Set up hand detection 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set up directory 
folder_path = '/Users/trangdang/Desktop/University of Chicago/5. Fall 2023/Capstone/Capstone II/data/train'
folder_list = os.listdir(folder_path)
folder_list = [item for item in folder_list if item != '.DS_Store']

# Initialize list to store input and output 
data = []
labels = []

dir_list = sorted(folder_list)

for dir_ in dir_list:
    count = 0
    
    # Retrieve list of image path in each class folder 
    img_path_list = os.listdir(os.path.join(folder_path, dir_))
    img_path_list = sorted(img_path_list)
    
    for img_path in img_path_list:
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(folder_path, dir_, img_path))
        
        # Check if image is invalid 
        if img is None:
            print(img_path)
        else:
            # Update image counter 
            count += 1
            
            # Convert image to RGB for Mediapipe to handle 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            # Process hands
            results = hands.process(img)
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                # Default hand detected at index 0
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Check in case multiple hands are identified 
                if len(results.multi_hand_landmarks) > 1:
                    hand_type = results.multi_handedness
                    
                    # Set reference for comparison 
                    index_ref = 0
                    score_ref = hand_type[index_ref].classification[0].score
                    
                    # Look at each hand identified
                    for i in range (len(hand_type)):
                        hand_type_score = hand_type[i].classification[0].score
                        
                        # Use the hand with highest confidence score
                        if hand_type_score > score_ref:
                            index_ref = i
                            score_ref = hand_type_score
                    
                    hand_landmarks = results.multi_hand_landmarks[index_ref]
                    
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