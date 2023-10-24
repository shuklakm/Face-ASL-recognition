#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:30:41 2023

@author: trangdang
"""

import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt

# Set up hand detection 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set up directory 
folder_path = './data'
folder_list = os.listdir(folder_path)
folder_list = [item for item in folder_list if item != '.DS_Store']

# Look at image instance
image_path = folder_list[1]
img = cv2.imread(os.path.join(folder_path, image_path))

# Convert to RGB for Mediapipe to process
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process hand 
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
            if hand_type_score > score_ref:
                index_ref = i
                score_ref = hand_type_score
        
        hand_landmarks = results.multi_hand_landmarks[index_ref]

    # for i in range (n_hands): 
    #     hand_side = results.multi_handedness[i]
    #     hand_landmarks = results.multi_hand_landmarks[i]
    # # for hand_landmarks in results.multi_hand_landmarks:
        
    #     print(f"This is hand no: {i}")
    #     print(hand_side)
        
    #     # Print the landmarks of each hand
    #     print(hand_landmarks)
        
    #     # Draw landmarks
    #     mp_drawing.draw_landmarks(
    #                 img, 
    #                 hand_landmarks, 
    #                 mp_hands.HAND_CONNECTIONS, 
    #                 mp_drawing_styles.get_default_hand_landmarks_style(), 
    #                 mp_drawing_styles.get_default_hand_connections_style())
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.show()
        
    #     print("-" * 20)