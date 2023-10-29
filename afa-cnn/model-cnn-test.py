import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Specify the path to the 'test1' folder
test_folder = './data/test1'

# Load the pre-trained CNN model
model = load_model('cnn-smnist100.h5')

# Create empty lists to store images, true labels (characters), and filenames
test_images = []
test_true_labels = []
test_filenames = []
test_predicted_labels = []  # Add a list to store predicted labels

# Define a mapping from filenames to characters (assuming the filename pattern 'X_test.jpg')
filename_to_character = lambda filename: filename.split('_')[0].upper()

# Iterate through the images in the 'test1' folder
for filename in os.listdir(test_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read and preprocess the image
        img = cv2.imread(os.path.join(test_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))  # Resize to match the model input shape
        img = img / 255.0  # Normalize pixel values
        img = img.reshape(28, 28, 1)  # Reshape to match model input shape

        # Extract the true label (character) from the filename
        true_label = filename_to_character(filename)

        # Append the preprocessed image, true label, and filename to the lists
        test_images.append(img)
        test_true_labels.append(true_label)
        test_filenames.append(filename)

# Convert the lists to NumPy arrays
test1_images = np.array(test_images)

# Make predictions on the 'test1' images using the loaded model
predictions = model.predict(test1_images)

# Assuming you have a labels_dict for mapping class indices to characters
labels_dict = {0: 'A', 1: 'B', 2: 'C',
               3: 'D', 4: 'E', 5: 'F',
               6: 'G', 7: 'H', 8: 'I',
               9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O',
               15: 'P', 16: 'Q', 17: 'R',
               18: 'S', 19: 'T', 20: 'U',
               21: 'V', 22: 'W', 23: 'X',
               24: 'Y', 25: 'Z'}

# Process the predictions and print the results
for i, prediction in enumerate(predictions):
    predicted_class = np.argmax(prediction)
    predicted_character = labels_dict[predicted_class]
    true_character = test_true_labels[i]
    confidence = prediction[predicted_class]

    # Append the predicted character to the list
    test_predicted_labels.append(predicted_character)

    print(f"File: {test_filenames[i]}, True Character: {true_character}, Predicted Character: {predicted_character}, Confidence: {confidence}")


# Calculate the confusion matrix
confusion = confusion_matrix(test_true_labels, test_predicted_labels)

# Calculate accuracy
accuracy = np.trace(confusion) / float(np.sum(confusion))
print("Accuracy:", accuracy)
