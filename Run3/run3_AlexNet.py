from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import optimizers, regularizers
import tensorflow as tf
from tensorflow.keras import layers, models
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
print("Average Width:", average_width)

# Preprocess images and create feature vectors
processed_images_input_layer = [process_image_alexnet(img_path, average_height, average_width) for img_path in
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

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
    create_alexnet_custom_lr = alex_net(input_shape= (average_height, average_width, 1))
    # create_alexnet_custom_lr = alex_net(input_shape=(224, 224, 1))

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