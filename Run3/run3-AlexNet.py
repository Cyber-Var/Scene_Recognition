from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from functions.functions import *
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

script_dir = os.path.dirname(os.path.realpath(__file__))

dataset_path = os.path.join(script_dir, "..", "training")
links, labels = load_dataset(dataset_path)

# Preprocess images and create feature vectors
feature_vectors = [Run3preprocess_image(i) for i in links]

# Print the shapes of the preprocessed images
for i, img in enumerate(feature_vectors):
    print(f"Shape of preprocessed image {i + 1}: {img.shape}")

# # Use label encoding to convert string labels to numerical format
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
# based in
# Cheng, G., Han, J., & Lu, X. (2017). Remote Sensing Image Scene Classification: Benchmark and State of the Art. In Proceedings of the IEEE (Vol. 105, Issue 10, pp. 1865–1883). Institute of Electrical and Electronics Engineers Inc. https://doi.org/10.1109/JPROC.2017.2675998
# paper
iterations = 15000
batch_size = 128
learning_rate = 0.001
last_layer_learning_rate = 0.01
weight_decay = 0.0005
momentum = 0.9

# Hyperparameters
# iterations = 50  # Reduce the number of iterations for faster training
# batch_size = 128  # Increase the batch size for faster computation
# learning_rate = 0.01
# last_layer_learning_rate = 0.1
# weight_decay = 0.005  # Slightly decrease weight decay
# momentum = 0.9

# iterations = 50
# batch_size = 32  # Smaller batch size can help with stability
# learning_rate = 0.001  # Lower learning rate to prevent overshooting
# last_layer_learning_rate = 0.001  # Keep the last layer learning rate similar to the base learning rate
# weight_decay = 1e-5  # Reduce weight decay to avoid excessive regularization
# momentum = 0.9


# AlexNet model with custom learning rates
def create_alexnet_custom_lr(num_classes=15):
    model = models.Sequential()

    # Layer 1
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 12)))

    # Layer 2
    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Layer 3
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

    # Layer 4
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

    # Layer 5
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Define custom learning rates for different layers
    # optimizer_last_layer = tf.keras.optimizers.legacy.SGD(learning_rate=last_layer_learning_rate, momentum=momentum, nesterov=True)
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)

    # Compile the model with custom learning rates
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if your labels are one-hot encoded
                  metrics=['accuracy'])

    # Set the last layer's learning rate
    model.layers[-1].set_weights([last_layer_learning_rate * w for w in model.layers[-1].get_weights()])

    return model

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Instantiate the model
alexnet_custom_lr = create_alexnet_custom_lr()

# Build the model
alexnet_custom_lr.build((None, 224, 224, 1))  # Specify input shape

# Display the model summary
alexnet_custom_lr.summary()

# Callbacks for model checkpointing and early stopping
model_checkpoint = ModelCheckpoint(filepath='alexnet_custom_lr_best_model.h5', save_best_only=True)
early_stopping = EarlyStopping(patience=2000, restore_best_weights=True)

# Train the model
with tf.device('/GPU:0'):
    history = alexnet_custom_lr.fit(X_train, y_train, epochs=iterations, batch_size=batch_size,
                                    validation_data=(X_test, y_test))

# Evaluate the model on the test data and print accuracy
test_loss, test_accuracy = alexnet_custom_lr.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# Predict probabilities on the test set using the features
y_probabilities = alexnet_custom_lr.predict(X_test)

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
