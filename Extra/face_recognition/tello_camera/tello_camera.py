from djitellopy import Tello
import cv2
import os
import time

# Ensure the directory exists
save_path = "./tello_camera/data"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Initialize the Tello drone
tello = Tello('192.168.87.58')
tello.connect()

# Start the video stream from the Tello drone
tello.streamon()

# Time delay for the camera to initialize and stabilize
time.sleep(2)

# Capture 10 images, for example
for i in range(41, 51):  # start from 121, end at 130
    frame = tello.get_frame_read().frame
    
    if frame is not None:
        # Convert the frame to RGB format for correct color display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save the frame as an image in the specified directory
        filename = os.path.join(save_path, f"Monika_{i}.jpg")
        cv2.imwrite(filename, frame_rgb)  # Saving the RGB frame
        print(f"Saved image {filename}")
        
        # Wait 2 seconds between captures (you can adjust this delay as needed)
        time.sleep(2)

# Cleanup
tello.streamoff()
