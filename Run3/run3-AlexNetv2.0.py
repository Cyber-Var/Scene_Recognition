import os
import time

import cv2
from statistics import mode
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from functions.functions import *
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

def read_images(file_path, name):
    try:
        images_map = []

        if name == "training":
            classes = os.listdir(file_path)
            for label in classes:
                label_folder = os.path.join(file_path, label)
                if os.path.isdir(label_folder):
                    images = os.listdir(os.path.join(file_path, label))

                    for image_name in images:
                        if image_name.lower().endswith(('.jpg', '.jpeg')):
                            image_file = os.path.join(label_folder, image_name)
                            images_map.append((cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), label))

        elif name == "testing":
            images = os.listdir(file_path)

            for image_name in images:
                if image_name.lower().endswith(('.jpg', '.jpeg')):
                    image_file = os.path.join(file_path, image_name)
                    images_map.append((cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), image_name.lower()))

        return images_map
    except Exception as e:
        print(f'Error when reading {name} data', e)


def check_size_and_scale(data, name):
    null = 0
    heights = []
    widths = []
    maxs = []
    mins = []
    scales = []
    for image in data:
        if image is not None:
            heights.append(image.shape[0])
            widths.append(image.shape[1])
            mins.append(image.min())
            maxs.append(image.max())
            scales.append(image.max() - image.min())
        else:
            null += 1

    print(f"{name} set size and scale information:")
    print("Number of images of type None =", null)
    print(f"Height: mode = {mode(heights)}, min = {min(heights)}, max = {max(heights)}")
    print(f"Width: mode = {mode(widths)}, min = {min(widths)}, max = {max(widths)}")
    print(f"Scale: average min = {mode(mins)}, average max = {mode(maxs)}, average scale range = {mode(scales) + 1} \n")


def split_image_into_patches(image, patch_size, sample_frequency):
    patches = []

    rows_max = image.shape[0] - patch_size
    cols_max = image.shape[1] - patch_size

    for row in range(0, rows_max + 1, sample_frequency):
        for col in range(0, cols_max + 1, sample_frequency):
            patch = image[row: row + patch_size, col: col + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)

    return np.array(patches)


def flatten_patches_into_vector(patches):
    flattened = np.array([patch.flatten() for patch in patches])

    mean_centered = flattened - np.mean(flattened, keepdims=True, axis=1)

    l2_norms_list = np.linalg.norm(mean_centered, keepdims=True, axis=1)
    l2_norms_list = np.where(l2_norms_list == 0, 1, l2_norms_list)
    normalized = mean_centered / l2_norms_list

    return normalized

training_data = read_images(os.path.join("..", "training"), "training")
testing_data = read_images(os.path.join("..", "testing"), "testing")
testing_filenames = [filename for image, filename in testing_data]
check_size_and_scale([image for image, label in training_data], "Training")
check_size_and_scale([image for image, filename in testing_data], "Testing")
#
# # TODO: maybe try different patch_size and sample_frequency, other than 8 and 4 ?
# training_patches = [(split_image_into_patches(image, 8, 4), label) for image, label in training_data]
# testing_patches = [split_image_into_patches(image, 8, 4) for image, filename in testing_data]
#
#
# training_vectors = [(flatten_patches_into_vector(patches), label) for patches, label in training_patches]
# testing_vectors = [flatten_patches_into_vector(patches) for patches in testing_patches]
#
#
#
# iterations = 15000
# batch_size = 128
# learning_rate = 0.001
# last_layer_learning_rate = 0.01
# weight_decay = 0.0005
# momentum = 0.9
#
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
#
# X_train = np.vstack([features for features, label in training_vectors])
# y_train = LabelEncoder().fit_transform([label for features, label in training_vectors])
#
# X_test = np.vstack(testing_vectors)
#
# # Assuming that your patches have a size of 8x8, so you have 64 features for each patch
# input_shape = (8, 8, 1)
# num_classes = len(set(y_train))
#
# # Reshape the data to match the input shape of the AlexNet model
# X_train = X_train.reshape(-1, *input_shape)
# X_test = X_test.reshape(-1, *input_shape)
# # Function to create AlexNet with custom learning rates
# def create_alexnet_custom_lr(input_shape=(8, 8, 1), num_classes=15):
#     model = models.Sequential()
#
#     # Layer 1
#     model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=input_shape))
#     model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
#
#     # Layer 2
#     model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
#
#     # Layer 3
#     model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
#
#     # Layer 4
#     model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
#
#     # Layer 5
#     model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
#
#     # Flatten and fully connected layers
#     model.add(layers.Flatten())
#     model.add(layers.Dense(4096, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(4096, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(num_classes, activation='softmax'))
#
#     # Define custom learning rates for different layers
#     optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
#
#     # Compile the model with custom learning rates
#     model.compile(optimizer=optimizer,
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
# # Assuming you want to use the AlexNet model with custom learning rates
# alexnet_custom_lr = create_alexnet_custom_lr(input_shape, num_classes)
#
#
# # Check if GPU is available
# if tf.test.gpu_device_name():
#     print('GPU available:', tf.test.gpu_device_name())
# else:
#     print("No GPU detected. Switching to CPU.")
#
# # Train the model
# with tf.device('/GPU:0'):
#     history = alexnet_custom_lr.fit(X_train, y_train, epochs=iterations, batch_size=batch_size, validation_split=0.2,
#                                     callbacks=[
#                                         ModelCheckpoint('alexnet_custom_lr.h5', save_best_only=True),
#                                         EarlyStopping(patience=5, restore_best_weights=True)
#                                     ])
#
# # Evaluate the model on the test set
# y_pred = np.argmax(alexnet_custom_lr.predict(X_test), axis=1)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
#
# print(f'Test Accuracy: {accuracy:.4f}')
# print(f'Weighted Precision: {precision:.4f}')
