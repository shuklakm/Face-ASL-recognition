#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:32:01 2023

@author: trangdang
"""

import os
import pickle

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Set up hand detection 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set up directory 
folder_path = './data/test_data'
folder_list = os.listdir(folder_path)
folder_list = [item for item in folder_list if item != '.DS_Store']

multiple_hand_image_path = []

for folder in folder_list: 
    print(f"Start folder {folder}")
    image_list = os.listdir(os.path.join(folder_path, folder))
    
    for image in image_list:
        img = cv2.imread(os.path.join(folder_path, folder, image))
        
        if img is not None:
            # Convert to RGB for Mediapipe to process
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process hand
            results = hands.process(img)
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                multiple_hand_image_path.append(image)
    print(f"Done folder {folder}")