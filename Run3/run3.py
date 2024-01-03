from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import sys
rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)

from functions.functions import *
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder


script_dir = os.path.dirname(os.path.realpath(__file__))

dataset_path = os.path.join(script_dir, "..", "training")
links, labels = load_dataset(dataset_path)

feature_vectors = []


# Preprocess images and create feature vectors
feature_vectors = [Run3preprocess_image(i).flatten() for i in links[:500]]
# X = feature_vectors

# Print the shapes of the preprocessed images
for i, img in enumerate(feature_vectors):
    print(f"Shape of preprocessed image {i + 1}: {img.shape}")

# # Use label encoding to convert string labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels[:500])


# Convert to NumPy array
X = np.array(feature_vectors)
y = np.array(y, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)



def create_alexnet(input_shape=(1024,), num_classes=10):
    model = models.Sequential()

    # Reshape the flattened input to match the size of your preprocessed images
    model.add(layers.Reshape((32, 32, 1), input_shape=input_shape))

    # Layer 1
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

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
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# Instantiate the model
alexnet = create_alexnet()

# Display the model summary
alexnet.summary()

# Compile the model
alexnet.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if your labels are one-hot encoded
                metrics=['accuracy'])



alexnet.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test data and print accuracy
test_loss, test_accuracy = alexnet.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')