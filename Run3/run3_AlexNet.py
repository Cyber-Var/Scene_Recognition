import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import optimizers, regularizers
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.densenet import preprocess_input


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


def load_testing_dataset(testing_folder):
    """
        Function to load the testing dataset, extracting image paths.
        :param testing_folder: Folder containing testing images
        :return: List of image paths
    """
    image_paths = []

    for filename in os.listdir(testing_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(testing_folder, filename)
            image_path = os.path.normpath(image_path)
            image_paths.append(image_path)

    return image_paths


def get_image_size(image_path):
    """
        Function to get the size of an image.
        :param image_path: Path of the image
        :return: Tuple representing image size
    """
    with Image.open(image_path) as img:
        return img.size


def calculate_images_int_average_size(image_paths):
    """
        Function to calculate the average height and width of a list of images.
        :param image_paths: List of image paths
        :return: Tuple representing average height and width
    """
    # Initialize variables to accumulate total height and width
    total_height = 0
    total_width = 0

    # Iterate through each image path
    for image_path in image_paths:
        # Get the size of the current image
        image_size = get_image_size(image_path)

        # Assuming your image size consists of (height, width)
        height, width = image_size

        # Accumulate the height and width
        total_height += height
        total_width += width

    # Calculate the average height and width
    average_height = total_height / len(image_paths)
    average_width = total_width / len(image_paths)

    # Return the result
    return int(round(average_height)), int(round(average_width))


def resize_image(image, average_height, average_width):
    """
        Function to resize an image.
        :param image: Image to be resized
        :param average_height: Target height
        :param average_width: Target width
        :return: Resized image
    """
    # Resize the image, resize() takes (width, height) as input
    resized_img = image.resize((average_width, average_height))
    return resized_img


def process_image_densenet_3d(image_path, average_height, average_width):
    """
        Function to process an image for DenseNet model.
        :param image_path: Path of the image
        :param average_height: Target height
        :param average_width: Target width
        :return: Processed image for DenseNet
    """
    with Image.open(image_path) as img:
        resized_image = resize_image(img, average_height, average_width)
        image_np_array = np.array(resized_image)
        # convert the image to 3 channels, so it can be processed by the model
        return np.stack((image_np_array,) * 3, axis=-1)  # Convert single-channel to three identical channels


def zero_mean_normalize(image):
    """
        Function that receives the image and applies zero mean normalization and scaling the pixels to achieve a unit standard deviation.
        :param image: image
        :return the normalized image
    """
    mean_value = np.mean(image)
    std_value = np.std(image)
    normalized_image = (image - mean_value) / std_value
    return normalized_image


def process_image_alexnet(image_path, average_height, average_width):
    """
        Function to process an image for AlexNet model.
        :param image_path: Path of the image
        :param average_height: Target height
        :param average_width: Target width
        :return: Processed image for AlexNet
    """
    with Image.open(image_path) as img:
        resized_image = resize_image(img, average_height, average_width)
        image_np_array = np.array(resized_image)
        nor_img = zero_mean_normalize(image_np_array)
        return nor_img


# DATA PREPROCESSING

# Load Dataset
script_dir = os.path.dirname(os.path.realpath(__file__))
training_dataset_path = os.path.join(script_dir, "..", "training")
training_image_paths, training_labels = load_training_dataset(training_dataset_path)

testing_dataset_path = os.path.join(script_dir, "..", "testing")
testing_image_paths = load_testing_dataset(testing_dataset_path)

# print(training_image_paths)
# print(testing_image_paths)

# Example usage:
average_height, average_width = calculate_images_int_average_size(training_image_paths)
# testing_average_height, testing_average_width = calculate_images_int_average_size(testing_image_paths)

print("Average Height:", average_height)
print("Average Weight:", average_width)

# Preprocess images and create feature vectors
processed_images_input_layer = [process_image_alexnet(img_path, 224, 224) for img_path in
                                training_image_paths]
testing_processed_images_input_layer = [
    process_image_alexnet(img_path, 224, 224) for img_path in
    testing_image_paths]

# # Print the shapes of the preprocessed images
# for i, img in enumerate(processed_images_input_layer):
#     print(f"Shape of preprocessed image {i + 1}: {img.shape}")
#
# for i, img in enumerate(testing_processed_images_input_layer):
#     print(f"Shape of preprocessed image {i + 1}: {img.shape}")

# Use label encoding to convert string labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(training_labels)

# Convert to NumPy array
X = np.array(processed_images_input_layer)
y = np.array(y, dtype=np.int32)

# print(processed_images_input_layer[0])
# print("HERE")
# print(testing_processed_images_input_layer[0])
#
testing_set = np.array(testing_processed_images_input_layer)
#

# print(X)
# print("HERE2")
# print(testing_set)

# MODEL TRAINING

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)

# MODEL TRAINING

# Hyperparameters based on paper
epochs = 100
batch_size = 32
learning_rate = 0.0001
last_layer_learning_rate = 0.001
weight_decay = 1e-5
# weight_decay = 0.0005

# momentum = 0.9

# Checkpoint path
checkpoint_path = 'alexnet_best_model.h5'

# ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_loss',  # Monitor the validation loss
    mode='min',  # Save when the monitored quantity is minimized
    verbose=1  # Display informative messages
)


# Early stopping
# early_stopping = EarlyStopping(patience=10, restore_best_weights=True)


def alex_net(input_shape, num_classes=15):
    model = models.Sequential()

    # Layer 1
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())

    # Layer 2
    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())

    # Layer 3
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())

    # Layer 4
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())

    # Layer 5
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # Additional Dense Layer
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Optimizer and Learning Rate Scheduler
    custom_optimizer = optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=custom_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Train the model
with tf.device('/GPU:0'):
    # Instantiate the model
    create_alexnet_custom_lr = alex_net(input_shape=(224, 224, 1))

    # Build the model
    create_alexnet_custom_lr.build()

    # Display the model summary
    create_alexnet_custom_lr.summary()

    # Fit the model with the ModelCheckpoint callback
    fitted_model = create_alexnet_custom_lr.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[model_checkpoint]  # Add the ModelCheckpoint callback
    )

# MODEL EVALUATION

# Load the best model
best_model = models.load_model(checkpoint_path)

# Accuracy

# Evaluate the best model on the test data
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f'\nBest Model Test Accuracy: {test_accuracy * 100:.2f}%')
