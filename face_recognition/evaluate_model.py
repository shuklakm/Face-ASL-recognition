import face_recognition
import os
from PIL import Image
import numpy as np
from face_recognition import face_distance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import csv





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

def recognize_faces_in_image(unknown_image_path, known_names, known_face_encodings, tolerance=0.5):
    # Load the image
    unknown_image = face_recognition.load_image_file(unknown_image_path)

    # Find faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image, model="cnn")
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # print(f"Found {len(unknown_face_encodings)} face(s) in this photograph.")


    # Loop through each face found in the unknown image
    for index, face_encoding in enumerate(unknown_face_encodings):

        # Calculate the face distance between the unknown face and all the known faces
        distances = face_distance(known_face_encodings, face_encoding)
        # print(f"Distances for face {index + 1}: {distances}")

        # Find the known face that has the smallest distance to the unknown face
        best_match_index = np.argmin(distances)
        # print(f"Best match index: {best_match_index}")


        if distances[best_match_index] < tolerance:
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        # print(f"Found {name} in the photo with a distance of {distances[best_match_index]}")

        return name

def extract_label_from_filename(file_name):
    """
    Extracts the label from the filename. Assumes the label is the filename without the extension.
    For example, 'Monika.jpg' will return 'Monika'.
    """
    label, _ = os.path.splitext(file_name)
    return label

def recognize_faces_in_directory(unknown_pictures_folder, known_names, known_face_encodings):
    for file_name in os.listdir(unknown_pictures_folder):
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(unknown_pictures_folder, file_name)
            print(f"Processing {file_name}...")
            recognize_faces_in_image(file_path, known_names, known_face_encodings)


def calculate_accuracy(unknown_pictures_folder, known_names, known_face_encodings, tolerance=0.5, results_file='/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/final/results_hog.csv'):

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_images = 0
    correct_predictions = 0

    true_labels = []
    predicted_labels = []
    
    file_names = os.listdir(unknown_pictures_folder)

    for file_name in file_names:
        true_label = extract_label_from_filename(file_name)
        file_path = os.path.join(unknown_pictures_folder, file_name)
        
        # Ensure we are only processing image files
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        predicted_label = recognize_faces_in_image(file_path, known_names, known_face_encodings)
        total_images += 1

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

        if predicted_label != "Unknown" and predicted_label.lower() == true_label.lower():
            true_positives +=1
        elif predicted_label != "Unknown" and predicted_label.lower() != true_label.lower():
            false_positives += 1
        elif predicted_label == "Unknown" and true_label.lower() not in known_names:
            true_positives +=1
        elif predicted_label == "Unknown" and true_label.lower() in known_names:
            false_negatives +=1

        # Check if predicted_label is a string and compare
        # if predicted_label and predicted_label.lower() == true_label.lower():
        #     correct_predictions += 1

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=known_names + ["Unknown"])

    
    # Calculate and return accuracy
    accuracy = true_positives / total_images if total_images > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # accuracy = correct_predictions / total_images if total_images > 0 else 0
     # Open the results file in append mode and write the results

    file_exists = os.path.isfile(results_file)

    with open(results_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Tolerance', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])
        # Write the tolerance and the corresponding metrics

        if not file_exists:
            writer.writeheader()

        writer.writerow({'Tolerance': tolerance, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1_Score': f1_score})

    return accuracy, precision, recall, f1_score, cm



# Paths
known_people_folder = "/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/known_people"
unknown_pictures_folder = "/Users/kajalshukla/Desktop/Capstone_2/Face-ASL-recognition/face_recognition_dlib/unknown_pictures"

# Load known faces
known_names, known_face_encodings = load_known_faces(known_people_folder)

# Calculate accuracy
accuracy, precision, recall, f1_score, cm = calculate_accuracy(unknown_pictures_folder, known_names, known_face_encodings)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

