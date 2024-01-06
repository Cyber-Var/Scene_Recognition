import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np


def load_training_dataset(root_folder):
    """
        Function to load the training dataset, extracting image paths and labels.
        :param root_folder: Root folder of the training dataset
        :return: Tuple of image paths and corresponding labels
    """
    image_paths = []
    labels = []

    for class_label, class_name in enumerate(os.listdir(root_folder)):
        class_folder = os.path.join(root_folder, class_name)

        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(class_folder, filename)
                    image_path = os.path.normpath(image_path)

                    image_paths.append(image_path)
                    labels.append(class_name)

    return image_paths, labels


script_dir = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(script_dir, "..", "training")
links, labels = load_training_dataset(dataset_path)

# Convert string labels to integer format
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(labels)


def create_dense_sift(image_path, target_size, step_size):
    # Read the grayscale image
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to the target size
    gray_image_resized = cv2.resize(gray_image, target_size[::-1])  # Reverse target_size for (width, height)

    # Create dense grid of keypoints
    keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray_image_resized.shape[0], step_size) for x in
                 range(0, gray_image_resized.shape[1], step_size)]

    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Compute SIFT descriptors for the dense keypoints
    keypoints, descriptors = sift.compute(gray_image_resized, keypoints)

    # Reshape the descriptors to form the dense SIFT image
    sift_image = descriptors.reshape((descriptors.shape[0], -1, 128))

    return sift_image


def create_sift_image(image_path, target_size):
    # Read the image turn grayscale if its not
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to the target size 
    gray_image_resized = cv2.resize(gray_image, target_size[::-1])  # Reverse target_size for (width, height)

    # Create the SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute SIFT descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image_resized, None)

    # Draw keypoints on the image
    sift_image = cv2.drawKeypoints(gray_image_resized, keypoints, None)

    return sift_image


# Testing Random Forests, KNN, Bayes, SVC, CNN we go the highest accuracy with SVC using the poly kernel with 8 degrees.
# Using 24 pixels as the dense parameter also gave us the highest accuracy when doing cross validation
# Dense Sift gave us way better results than normal sift, therefore we will be using dense sift to pre process the images. 


sift_images = []

# Pre Processing each image with dense sift
for i in links:
    sift_img = create_dense_sift(i, (224, 224), 24)
    sift_images.append(sift_img)

# Flatten the dense SIFT images
flattened_sift_images = [sift_img.flatten() for sift_img in sift_images]

# data splitting
X_train, X_test, y_train, y_test = train_test_split(flattened_sift_images, numerical_labels, test_size=0.2,
                                                    random_state=42)

svm_classifier = SVC(kernel='poly', decision_function_shape='ovr', degree=8)

# train
svm_classifier.fit(X_train, y_train)

# Predict the test 
y_pred = svm_classifier.predict(X_test)

# acuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
