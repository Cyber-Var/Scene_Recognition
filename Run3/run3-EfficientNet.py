from keras.src.applications import EfficientNetB0
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from functions.functions import *
import os
import numpy as np

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# Hyperparameters
iterations = 15000
batch_size = 128
learning_rate = 0.001
weight_decay = 0.0005

# Function to create the model with EfficientNet as the base
def create_advanced_efficientnet_model(input_shape=(224, 224, 3), num_classes=15, learning_rate=0.001):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')

    # Fine-tune the last few layers of the base model
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    custom_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    return lr

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Instantiate the improved model with EfficientNet
advanced_efficientnet_model = create_advanced_efficientnet_model(input_shape=(224, 224, 3))

# Build the model
advanced_efficientnet_model.build((None, 224, 224, 3))  # Specify input shape

# Display the model summary
advanced_efficientnet_model.summary()

# Callbacks for model checkpointing, early stopping, and learning rate scheduling
model_checkpoint = ModelCheckpoint(filepath='advanced_efficientnet_model_best_model.h5', save_best_only=True)
early_stopping = EarlyStopping(patience=200, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the improved model
with tf.device('/GPU:0'):
    history = advanced_efficientnet_model.fit(X_train, y_train, epochs=iterations, batch_size=batch_size,
                                              validation_data=(X_test, y_test),
                                              callbacks=[model_checkpoint, early_stopping, lr_scheduler])

# Evaluate the improved model on the test data and print accuracy
test_loss, test_accuracy = advanced_efficientnet_model.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# Predict probabilities on the test set using the features
y_probabilities = advanced_efficientnet_model.predict(X_test)

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

plt.legend()
plt.show()
