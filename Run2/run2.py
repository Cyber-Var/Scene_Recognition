import os
import time
import cv2
from statistics import mode
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.svm import SVC


def read_images(file_path, name):
    """
        Function that reads images from files into numpy arrays
        :param file_path: Path to folder containing images
        :param name: either training or testing
        :return list of pairs in the form (image array, label) for training set and (image array, filename)
                for testing set
    """
    try:
        images_list = []

        # If reading files from training dataset:
        if name == "training":
            # Loop through folders corresponding to each class:
            classes = os.listdir(file_path)
            for label in classes:
                label_folder = os.path.join(file_path, label)
                if os.path.isdir(label_folder):
                    # Loop through images inside each folder:
                    images = os.listdir(os.path.join(file_path, label))
                    for image_name in images:
                        if image_name.lower().endswith(('.jpg', '.jpeg')):
                            # Write the image and its label to the list of pairs:
                            image_file = os.path.join(label_folder, image_name)
                            images_list.append((cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), label))

        # If reading files from testing dataset:
        elif name == "testing":
            images = os.listdir(file_path)

            # Loop through every image in the testing dataset:
            for image_name in images:
                if image_name.lower().endswith(('.jpg', '.jpeg')):
                    # Write the image and its filename to the list of pairs:
                    image_file = os.path.join(file_path, image_name)
                    images_list.append((cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), image_name.lower()))

        return images_list
    # Raise exception if there occurs a problem with reading files:
    except Exception as e:
        print(f'Error when reading {name} data', e)


def check_size_and_scale(data, name):
    """
        Function that prints size and scale information about images
        :param data: list of images in the form of numpy arrays
        :param name: either Training or Testing (used for printing information)
    """
    null = 0
    heights = []
    widths = []
    maxs = []
    mins = []
    scales = []

    # Loop through every image in the given list:
    for image in data:
        if image is not None:
            # Store the size and scale information of each image:
            heights.append(image.shape[0])
            widths.append(image.shape[1])
            mins.append(image.min())
            maxs.append(image.max())
            scales.append(image.max() - image.min())
        else:
            # Count the number of images of type None, if there are any:
            null += 1

    # Print the overall size and scale information of the dataset:
    print(f"{name} set size and scale information:")
    print("Number of images of type None =", null)
    print(f"Height: mode = {mode(heights)}, min = {min(heights)}, max = {max(heights)}")
    print(f"Width: mode = {mode(widths)}, min = {min(widths)}, max = {max(widths)}")
    print(f"Scale: average min = {mode(mins)}, average max = {mode(maxs)}, average scale range = {mode(scales) + 1} \n")


def split_image_into_patches(image, patch_size, sample_frequency):
    """
        Function that splits an image into patches
        :param image: the image to split
        :param patch_size: the resulting patches will be N*N, where N is the patch_size
        :param sample_frequency: image will be sampled every N pixels in x and y directions,
                                 where N is the sample_frequency
        :return list of patches for the image
    """
    patches = []

    rows_max = image.shape[0] - patch_size
    cols_max = image.shape[1] - patch_size

    for row in range(0, rows_max + 1, sample_frequency):
        for col in range(0, cols_max + 1, sample_frequency):
            # Extract a patch from the image:
            patch = image[row: row + patch_size, col: col + patch_size]
            # Ignore right and bottom patches that do not result in desired size:
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # Store the patch in the list of patches:
                patches.append(patch)

    return np.array(patches)


def normalize_and_mean_center_patches(patches):
    """
        Function that mean-centers and normalizes patches
        :param patches: list of patches of an image
        :return list of mean-centered and normalized patches
    """
    # Mean-center each patch:
    mean_centered = patches - np.mean(patches, keepdims=True, axis=(1, 2))

    # Normalize each mean-centered patch:
    l2_norms_list = np.array([np.linalg.norm(p, keepdims=True) for p in mean_centered])
    l2_norms_list = np.where(l2_norms_list == 0, 1, l2_norms_list)
    mean_centered_normalized = mean_centered / l2_norms_list

    return mean_centered_normalized


def flatten_patches_into_vector(patches):
    """
        Function that flattens patches of an image into vectors
        :param patches: list of patches of an image
        :return list flattened vectors
    """

    # Flatten the patches into vectors:
    flattened = np.array([patch.flatten() for patch in patches])
    return flattened


def learn_vocabulary(data):
    """
        Function that performs K-Means clustering and to create the visual vocabulary
        :param data: list of flattened vectors of an image
        :return the learnt visual vocabulary
    """

    # Prepare the data for clustering:
    clustering_data = [vector for vector, label in data]
    clustering_data_prepared = [patch for vector in clustering_data for patch in vector]

    # Perform the K-Means clustering:
    k_means = MiniBatchKMeans(n_clusters=150, n_init=10, batch_size=1000, random_state=0)
    k_means.fit(clustering_data_prepared)

    # Return the learnt visual vocabulary:
    return k_means.cluster_centers_


def vector_quantisation(vectors, vocab):
    """
        Function that maps each feature vector to the nearest visual word
        :param vectors: list of feature vectors
        :param vocab: vocabulary learnt from training set's images
        :return list of visual words
    """

    words = []

    for vector in vectors:
        # Find the words closest to each feature vector:
        distances = np.linalg.norm(vocab - vector, axis=1)
        words.append(np.argmin(distances))

    return words


def train_classifiers(input_features, targets):
    """
        Function that trains a one-vs-all classifier for each class
        :param input_features: list of training input features
        :param targets: list of correct training labels
        :return list of classifiers for each class
    """
    classifiers_for_each_label = []
    classes = set(targets)

    # Loop through each class type:
    for i in range(len(classes)):
        # Set only the current class as the positive class:
        class_targets = [1 if target == i else 0 for target in targets]

        # Create and train the Support Vector Machines model:
        svm = SVC(kernel='linear', probability=True, random_state=42)
        svm.fit(input_features, class_targets)

        # Add the trained model to the list of classifiers:
        classifiers_for_each_label.append(svm)

    return classifiers_for_each_label


def make_predictions(input_features, classifiers_for_each_label):
    """
        Function that receives predictions from each classifier and combines them to determine the most likely class
        :param input_features: list of testing input features
        :param classifiers_for_each_label: list of trained classifiers for each class
        :return final predictions
    """
    predictions_for_each_class = np.zeros((len(input_features), len(classifiers_for_each_label)))

    # Predict probabilities of whether features belong to each class:
    for i, classifier in enumerate(classifiers_for_each_label):
        predictions_for_each_class[:, i] = classifier.predict_proba(input_features)[:, 1]

    # Return the list of prediction classes with maximum probability:
    return np.argmax(predictions_for_each_class, axis=1)


def write_predictions_to_file(predicted, file_names, label_encoder):
    """
        Function that writes final predictions to the run2.txt file, as instructed in specification
        :param predicted: list of predictions
        :param file_names: list of all file names from the testing dataset
        :param label_encoder: LabelEncoder object that was used to encode classes (will be used to decode labels
                              back to string format)
    """
    # Transform encoded labels back to string:
    string_predictions = label_encoder.inverse_transform(predicted)

    # Write predictions and file names to the run2.txt file:
    with open("run2.txt", 'w') as output_file:
        for filename, prediction in zip(file_names, string_predictions):
            output_file.write(f"{filename} {prediction}\n")


# Read training and testing data from files:
training_data = read_images(os.path.join("..", "training"), "training")
testing_data = read_images(os.path.join("..", "testing"), "testing")
testing_filenames = [filename for image, filename in testing_data]

# Print information about size and scale of images in datasets:
check_size_and_scale([image for image, label in training_data], "Training")
check_size_and_scale([image for image, filename in testing_data], "Testing")


# Split the images into 5 by 5 patches, sampled every 5 pixels in the x and y directions:
training_patches = [(split_image_into_patches(image, 7, 4), label) for image, label in training_data]
testing_patches = [split_image_into_patches(image, 7, 4) for image, filename in testing_data]

# Mean-center and normalize patches:
training_mean_centered_normalized = [(normalize_and_mean_center_patches(patches), label) for patches, label in training_patches]
testing_mean_centered_normalized = [normalize_and_mean_center_patches(patches) for patches in testing_patches]

# Flatten patches into vectors:
training_vectors = [(flatten_patches_into_vector(patches), label) for patches, label in training_mean_centered_normalized]
testing_vectors = [flatten_patches_into_vector(patches) for patches in testing_mean_centered_normalized]


# Apply K-Means clustering on training data to learn the visual vocabulary:
vocabulary = learn_vocabulary(training_vectors)

# Map each feature vector to the closest visual word using vocabulary learnt from training data:
training_quantized_words = [vector_quantisation(vector, vocabulary) for vector, label in training_vectors]
testing_quantized_words = [vector_quantisation(vector, vocabulary) for vector in testing_vectors]

# Draw a histogram over the visual word counts for each image:
bins = np.arange(len(vocabulary+1))
training_feature_vectors = [np.histogram(word, bins=bins, density=True)[0] for word in training_quantized_words]
testing_feature_vectors = [np.histogram(word, bins=bins, density=True)[0] for word in testing_quantized_words]


# Encode labels into numerical form:
labels = [label.lower() for vector, label in training_vectors]
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)


# Split the training dataset (80% for training and 20% for testing) to evaluate the model's performance:
X_train, X_test, y_train, y_test = train_test_split(training_feature_vectors, labels_encoded, test_size=0.2,
                                                    random_state=42, stratify=labels_encoded)
classifiers_for_precision_score = train_classifiers(X_train, y_train)
predictions_for_precision_score = make_predictions(X_test, classifiers_for_precision_score)
accuracy = accuracy_score(y_test, predictions_for_precision_score)
precision = precision_score(y_test, predictions_for_precision_score, average='macro', zero_division=0)
print("Accuracy:", accuracy)
print("Average Precision:", precision)


# Train a new model on the full training dataset:
one_vs_all_classifiers = train_classifiers(training_feature_vectors, labels_encoded)

# Make classification predictions on the testing set and write the results to run2.txt:
final_predictions = make_predictions(testing_feature_vectors, one_vs_all_classifiers)
write_predictions_to_file(final_predictions, testing_filenames, encoder)
