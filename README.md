# Drone-based ASL Recognition with Facial Locking
### Description
This project aims to provide a solution for real-time American Sign Language (ASL) recognition using drones. By employing the YOLOv5 model, the drone's camera identifies and locks onto a signer's face, ensuring that hand gestures remain consistently in frame for subsequent ASL gesture analysis.

The system promises a transformative solution for the deaf community, enabling more seamless communication in diverse scenarios like travel, search and rescue operations, and more.

### Features
Real-time face detection using YOLOv5.
Face tracking to keep signer's face (and hands) centered.
Integration with ASL gesture recognition (future scope).

### Requirements
Python 3.x
OpenCV
Torch
djitellopy (for Tello drone integration)
