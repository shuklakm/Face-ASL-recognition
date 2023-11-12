import cv2
import torch
from djitellopy import Tello

# Paths
# weights_path = "/Users/kajalshukla/Desktop/ASL/yolov5/runs/train/exp16/weights/best.pt" Good One! 

weights_path = "/Users/kajalshukla/Desktop/ASL/yolov5/runs/train/exp18/weights/best.pt"


# Setup YOLOv5 with custom model weights
model = torch.hub.load('./yolov5', 'custom', path=weights_path, source='local')

# Initialize the Tello drone
tello = Tello('192.168.87.58')
tello.connect()
tello.streamon()

while True:
    # Get the current frame from the Tello drone video stream
    frame = tello.get_frame_read().frame
    if frame is None:
        print("Failed to grab frame")
        break

    # Display raw Tello frame
    cv2.imshow('Raw Tello Frame', frame)

    # Assuming Tello sends frames in RGB format, convert to RGB for YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pass the frame to YOLOv5 for object detection with confidence threshold set to 0.10
    model.conf = 0.10
    results = model(frame_rgb)

    # Convert results to rendered frame and display
    rendered_frame = results.render()[0]
    rendered_frame_bgr = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLOv5 Tello', rendered_frame_bgr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
tello.streamoff()
cv2.destroyAllWindows()

