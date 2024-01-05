from PIL import Image
import numpy as np
from keras.src.applications import DenseNet201
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.densenet import preprocess_input

# from functions.functions import *
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder



def run3Resize_img(image):
    resized_img = image.resize((224, 224))
    return resized_img

def load_dataset(root_folder):
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

def Run3preprocess_image(image_path):
    with Image.open(image_path) as img:
        img_sized = run3Resize_img(img)
        img_array = np.array(img_sized)
        return np.stack((img_array,) * 3, axis=-1)  # Convert single-channel to three identical channels



script_dir = os.path.dirname(os.path.realpath(__file__))

dataset_path = os.path.join(script_dir, "..", "training")
links, labels = load_dataset(dataset_path)

# Preprocess images and create feature vectors
feature_vectors = [Run3preprocess_image(i) for i in links]

# Print the shapes of the preprocessed images
for i, img in enumerate(feature_vectors):
    print(f"Shape of preprocessed image {i + 1}: {img.shape}")

# Use label encoding to convert string labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Convert to NumPy array
X = np.array(feature_vectors)
y = np.array(y, dtype=np.int32)

# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU available:', tf.test.gpu_device_name())
else:
    print("No GPU detected. Switching to CPU.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)
# Hyperparameters based on paper
iterations = 10
batch_size = 32
learning_rate = 0.001
last_layer_learning_rate = 0.01
weight_decay = 0.0005
momentum = 0.9

def create_densenet(input_shape=(224, 224, 3), num_classes=15, learning_rate=0.001):
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),  # Add a dense layer with fewer neurons
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    custom_optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=custom_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# # Instantiate DenseNet model
# densenet_model = create_densenet(input_shape=(224, 224, 3), num_classes=15, learning_rate=learning_rate

# Train the improved model
with tf.device('/GPU:0'):
    # Build the model
    densenet_model = create_densenet(input_shape=(224, 224, 3), num_classes=15, learning_rate=learning_rate)

    # Display the model summary
    densenet_model.summary()

    # Callbacks for model checkpointing and early stopping
    model_checkpoint = ModelCheckpoint(filepath='densenet_model_best_model.h5', save_best_only=True)
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    # Train the model on GPU
    history = densenet_model.fit(preprocess_input(X_train), y_train, epochs=iterations, batch_size=batch_size,
                                 validation_data=(preprocess_input(X_test), y_test),
                                 callbacks=[model_checkpoint, early_stopping])


# Evaluate the model on the test data and print accuracy
test_loss, test_accuracy = densenet_model.evaluate(preprocess_input(X_test), y_test)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# Predict probabilities on the test set using the features
y_probabilities = densenet_model.predict(preprocess_input(X_test))

# Calculate Average Precision for each class
average_precisions = []
for class_index in range(15):
    true_labels_class = (y_test == class_index).astype(int)
    predicted_scores_class = y_probabilities[:, class_index]

    # Compute average precision for the current class
    average_precision = average_precision_score(true_labels_class, predicted_scores_class)
    average_precisions.append(average_precision)

    print(f'Average Precision for class {class_index}: {average_precision:.4f}')

# Calculate and print the mean Average Precision (mAP)
mean_average_precision = np.mean(average_precisions)
print(f'Mean Average Precision (mAP): {mean_average_precision:.4f}')
