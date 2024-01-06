# DenseNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.src.applications import DenseNet201
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.densenet import preprocess_input

from functions.run3_functions import *

# DATA PREPROCESSING

# Load Dataset
script_dir = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(script_dir, "..", "training")
image_paths, labels = load_dataset(dataset_path)

# Example usage:
average_height, average_width = calculate_images_int_average_size(image_paths)

print("Average Height:", average_height)
print("Average Weight:", average_width)

# Preprocess images and create feature vectors
processed_images_input_layer = [process_image_densenet_3d(img_path, average_height, average_width) for img_path in
                                image_paths]

# Print the shapes of the preprocessed images
for i, img in enumerate(processed_images_input_layer):
    print(f"Shape of preprocessed image {i + 1}: {img.shape}")

# Use label encoding to convert string labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Convert to NumPy array
X = np.array(processed_images_input_layer)
y = np.array(y, dtype=np.int32)

# MODEL TRAINING

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)

# # Hyperparameters based on paper
# epochs = 10
# batch_size = 32
# learning_rate = 0.001
# last_layer_learning_rate = 0.01
# weight_decay = 0.0005
# momentum = 0.9

# Hyperparameters based on paper
epochs = 15
batch_size = 32
learning_rate = 0.0001
last_layer_learning_rate = 0.001
weight_decay = 1e-5
# weight_decay = 0.0005

# Checkpoint path
checkpoint_path = 'densenet_best_model.h5'

# ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_loss',  # Monitor the validation loss
    mode='min',  # Save when the monitored quantity is minimized
    verbose=2  # Display informative messages
)


def dense_net_201(input_shape=(average_height, average_width, 3), num_classes=15):
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    custom_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=custom_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU available:', tf.test.gpu_device_name())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("No GPU detected. Switching to CPU.")

# Train the model on GPU
with tf.device('/GPU:0'):
    # Build the model
    densenet_model = dense_net_201()

    # Display the model summary
    densenet_model.summary()

    # Train the model on GPU
    fitted_model = densenet_model.fit(preprocess_input(X_train), y_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=(preprocess_input(X_test), y_test),
                                      callbacks=[model_checkpoint])

# Load the best model
best_model = models.load_model(checkpoint_path)

# MODEL EVALUATION

# # Evaluate the model on the test data and print accuracy
test_loss, test_accuracy = best_model.evaluate(preprocess_input(X_test), y_test)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')


test_dir = os.path.join(script_dir, "..", "testing")

test_file_names = []
test_class = []

for i in range(0,2988):
    try:
        file_path_temp = os.path.join(test_dir, f"{i}.jpg")

        preprocessed_image = preprocess_image(file_path_temp).flatten()

        preprocessed_image = preprocessed_image.reshape(1, -1)

        predicted_class = knn_classifier.predict(preprocessed_image)

        test_class.append(predicted_class)
        test_file_names.append(str(i) + ".jpg")
    except Exception as e:
        print(f"Error processing image {file_path_temp}: {e}")


#Create the run1.txt file and write all the test classification
with open("run3.txt", 'w') as output_file:
    for m in range(0, len(test_class)):
        output_file.write(str(test_file_names[m]) + " " + str(test_class[m][0]) + "\n")
















# AlexNet

#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import average_precision_score
# from sklearn.preprocessing import LabelEncoder
#
# from tensorflow.keras import optimizers, regularizers
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.applications.densenet import preprocess_input
#
# from functions.run3_functions import *
#
# # DATA PREPROCESSING
#
# # Load Dataset
# script_dir = os.path.dirname(os.path.realpath(__file__))
# dataset_path = os.path.join(script_dir, "..", "training")
# image_paths, labels = load_dataset(dataset_path)
#
# # Example usage:
# average_height, average_width = calculate_images_int_average_size(image_paths)
#
# print("Average Height:", average_height)
# print("Average Width:", average_width)
#
# # Preprocess images and create feature vectors
# processed_images_input_layer = [process_image_alexnet(img_path, average_height, average_width) for img_path in
#                                 image_paths]
#
# # Print the shapes of the preprocessed images
# for i, img in enumerate(processed_images_input_layer):
#     print(f"Shape of preprocessed image {i + 1}: {img.shape}")
#
# # Use label encoding to convert string labels to numerical format
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(labels)
#
# # Convert to NumPy array
# X = np.array(processed_images_input_layer)
# y = np.array(y, dtype=np.int32)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
#
# # MODEL TRAINING
#
# # Hyperparameters based on paper
# epochs = 100
# batch_size = 32
# learning_rate = 0.0001
# last_layer_learning_rate = 0.001
# weight_decay = 1e-5
# # weight_decay = 0.0005
#
# # momentum = 0.9
#
# # Checkpoint path
# checkpoint_path = 'alexnet_best_model.h5'
#
# # ModelCheckpoint callback
# model_checkpoint = ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_best_only=True,
#     monitor='val_loss',  # Monitor the validation loss
#     mode='min',  # Save when the monitored quantity is minimized
#     verbose=1  # Display informative messages
# )
#
# # Early stopping
# # early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
#
#
# def alex_net(input_shape, num_classes=15):
#     model = models.Sequential()
#
#     # Layer 1
#     model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=input_shape,
#                             kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
#     model.add(layers.BatchNormalization())
#
#     # Layer 2
#     model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu',
#                             kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
#     model.add(layers.BatchNormalization())
#
#     # Layer 3
#     model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu',
#                             kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(layers.BatchNormalization())
#
#     # Layer 4
#     model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu',
#                             kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(layers.BatchNormalization())
#
#     # Layer 5
#     model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu',
#                             kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
#     model.add(layers.BatchNormalization())
#
#     # Global Average Pooling
#     model.add(layers.GlobalAveragePooling2D())
#
#     # Fully Connected Layers
#     model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(0.5))
#
#     model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(0.5))
#
#     # Additional Dense Layer
#     model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(0.5))
#
#     # Output Layer
#     model.add(layers.Dense(num_classes, activation='softmax'))
#
#     # Optimizer and Learning Rate Scheduler
#     custom_optimizer = optimizers.legacy.Adam(learning_rate=learning_rate)
#     model.compile(optimizer=custom_optimizer,
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
#
# # Check if GPU is available
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#
# # Train the model
# with tf.device('/GPU:0'):
#     # Instantiate the model
#     create_alexnet_custom_lr = alex_net(input_shape= (average_height, average_width, 1))
#     # create_alexnet_custom_lr = alex_net(input_shape=(224, 224, 1))
#
#     # Build the model
#     create_alexnet_custom_lr.build()
#
#     # Display the model summary
#     create_alexnet_custom_lr.summary()
#
#     # Fit the model with the ModelCheckpoint callback
#     fitted_model = create_alexnet_custom_lr.fit(
#         X_train, y_train, epochs=epochs, batch_size=batch_size,
#         validation_data=(X_test, y_test),
#         callbacks=[model_checkpoint]  # Add the ModelCheckpoint callback
#     )
#
# # MODEL EVALUATION
#
# # Load the best model
# best_model = models.load_model(checkpoint_path)
#
# # Accuracy
#
# # Evaluate the best model on the test data
# test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
# print(f'\nBest Model Test Accuracy: {test_accuracy * 100:.2f}%')
#
#



# SVM with dense SIFT

# import cv2
# import os
# from sklearn.neighbors import KNeighborsClassifier
# import sys
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# import numpy as np
#
#
# def load_dataset(root_folder):
#     image_paths = []
#     labels = []
#
#     for class_label, class_name in enumerate(os.listdir(root_folder)):
#         class_folder = os.path.join(root_folder, class_name)
#
#         if os.path.isdir(class_folder):
#             for filename in os.listdir(class_folder):
#                 if filename.endswith(".jpg"):
#                     image_path = os.path.join(class_folder, filename)
#                     image_path = os.path.normpath(image_path)
#
#                     image_paths.append(image_path)
#                     labels.append(class_name)
#
#     return image_paths, labels
#
#
# script_dir = os.path.dirname(os.path.realpath(__file__))
# dataset_path = os.path.join(script_dir, "..", "training")
# links, labels = load_dataset(dataset_path)
#
# # Convert string labels to integer format
# label_encoder = LabelEncoder()
# numerical_labels = label_encoder.fit_transform(labels)
#
#
# def create_dense_sift(image_path, target_size, step_size):
#     # Read the grayscale image
#     gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # Resize the image to the target size
#     gray_image_resized = cv2.resize(gray_image, target_size[::-1])  # Reverse target_size for (width, height)
#
#     # Create dense grid of keypoints
#     keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray_image_resized.shape[0], step_size) for x in
#                  range(0, gray_image_resized.shape[1], step_size)]
#
#     # Create a SIFT object
#     sift = cv2.SIFT_create()
#
#     # Compute SIFT descriptors for the dense keypoints
#     keypoints, descriptors = sift.compute(gray_image_resized, keypoints)
#
#     # Reshape the descriptors to form the dense SIFT image
#     sift_image = descriptors.reshape((descriptors.shape[0], -1, 128))
#
#     return sift_image
#
#
# def create_sift_image(image_path, target_size):
#     # Read the image turn grayscale if its not
#     gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # Resize the image to the target size
#     gray_image_resized = cv2.resize(gray_image, target_size[::-1])  # Reverse target_size for (width, height)
#
#     # Create the SIFT object
#     sift = cv2.SIFT_create()
#
#     # Detect keypoints and compute SIFT descriptors
#     keypoints, descriptors = sift.detectAndCompute(gray_image_resized, None)
#
#     # Draw keypoints on the image
#     sift_image = cv2.drawKeypoints(gray_image_resized, keypoints, None)
#
#     return sift_image
#
#
# # Testing Random Forests, KNN, Bayes, SVC, CNN we go the highest accuracy with SVC using the poly kernel with 8 degrees.
# # Using 24 pixels as the dense parameter also gave us the highest accuracy when doing cross validation
# # Dense Sift gave us way better results than normal sift, therefore we will be using dense sift to pre process the images.
#
#
# sift_images = []
#
# # Pre Processing each image with dense sift
# for i in links:
#     sift_img = create_dense_sift(i, (224, 224), 24)
#     sift_images.append(sift_img)
#
# # Flatten the dense SIFT images
# flattened_sift_images = [sift_img.flatten() for sift_img in sift_images]
#
# # data splitting
# X_train, X_test, y_train, y_test = train_test_split(flattened_sift_images, numerical_labels, test_size=0.2,
#                                                     random_state=42)
#
# svm_classifier = SVC(kernel='poly', decision_function_shape='ovr', degree=8)
#
# # train
# svm_classifier.fit(X_train, y_train)
#
# # Predict the test
# y_pred = svm_classifier.predict(X_test)
#
# # acuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
