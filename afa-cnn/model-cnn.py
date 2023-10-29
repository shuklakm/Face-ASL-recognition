import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
import cv2
import pandas as pd

# Load your custom dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Assuming 'data' contains your original dataset
# Resize each image to 28x28 pixels
resized_data = np.array([cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA) for img in data])

# Convert labels to one-hot encoding
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(resized_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Verify the shape of x_train before reshaping
print("x_train Shape Before Reshaping:", x_train.shape)

# Reshape x_train and x_test to match the expected input shape of the CNN model
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

num_classes = 26

model = Sequential()

# Adding layers to the model
# Layer 1: Convolutional layer with 75 filters, followed by BatchNormalization and MaxPooling
model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

# Freezing the initial layers
model.layers[0].trainable = False
model.layers[1].trainable = False
model.layers[2].trainable = False

# Adding more layers
# Layer 2: Convolutional layer with 50 filters, followed by Dropout, BatchNormalization, and MaxPooling
model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

# Layer 3: Convolutional layer with 25 filters, followed by BatchNormalization and MaxPooling
model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

# Flattening the output from the convolutional layers before passing them into Dense layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation='softmax'))  # Using softmax activation function for the output layer

# Compiling the model with 'adam' optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Displaying the model summary which shows the architecture, trainable parameters, etc.
model.summary()


# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

datagen.fit(x_train)

# Training the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=100, validation_data=(x_test, y_test))

# Save the trained model
model.save('cnn-smnist100v2.h5')

# Create a DataFrame to store the metrics per epoch
metrics_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),
    'Loss': history.history['loss'],
    'Accuracy': history.history['accuracy'],
    'Val_Loss': history.history['val_loss'],
    'Val_Accuracy': history.history['val_accuracy']
})

# Save the metrics to a CSV file
metrics_df.to_csv('cnn-metricsv2.csv', index=False)

# Evaluate the model and print metrics
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
