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
