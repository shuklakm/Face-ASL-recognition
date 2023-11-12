import face_recognition
import os
from PIL import Image
import numpy as np
from face_recognition import face_distance

def load_known_faces(known_people_folder):
    known_names = []
    known_face_encodings = []

    # Browse through each person in the known people folder
    for name in os.listdir(known_people_folder):
        subdir = os.path.join(known_people_folder, name)
        if os.path.isdir(subdir):
            person_encodings = []
            # Now go through each image of a person and get its face encoding
            for file_name in os.listdir(subdir):
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    file_path = os.path.join(subdir, file_name)

                    # Load each image file and get face encodings
                    image = face_recognition.load_image_file(file_path)
                    face_encodings = face_recognition.face_encodings(image)

                    if face_encodings:
                        # Add face encoding and name to our known faces
                        known_face_encodings.append(face_encodings[0])
                        known_names.append(name)
                    else:
                        print(f"No faces detected in the image {subdir} {file_name}!")


    return known_names, known_face_encodings

def recognize_faces_in_image(unknown_image_path, known_names, known_face_encodings, tolerance=0.4):
    # Load the image
    unknown_image = face_recognition.load_image_file(unknown_image_path)

    # Find faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image, model="cnn")
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    print(f"Found {len(unknown_face_encodings)} face(s) in this photograph.")


    # Loop through each face found in the unknown image
    for index, face_encoding in enumerate(unknown_face_encodings):

        # Calculate the face distance between the unknown face and all the known faces
        distances = face_distance(known_face_encodings, face_encoding)
        print(f"Distances for face {index + 1}: {distances}")

        # Find the known face that has the smallest distance to the unknown face
        best_match_index = np.argmin(distances)
        print(f"Best match index: {best_match_index}")

        if distances[best_match_index] < tolerance:
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        print(f"Found {name} in the photo with a distance of {distances[best_match_index]}")

def recognize_faces_in_directory(unknown_pictures_folder, known_names, known_face_encodings):
    for file_name in os.listdir(unknown_pictures_folder):
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(unknown_pictures_folder, file_name)
            print(f"Processing {file_name}...")
            recognize_faces_in_image(file_path, known_names, known_face_encodings)

# Paths
known_people_folder = "/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/known_people"
unknown_pictures_folder = "/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/unknown_pictures"

# Load known faces
known_names, known_face_encodings = load_known_faces(known_people_folder)

# Recognize faces in all unknown images
recognize_faces_in_directory(unknown_pictures_folder, known_names, known_face_encodings)