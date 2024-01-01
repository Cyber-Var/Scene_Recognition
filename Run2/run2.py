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


def learn_vocabulary(data):
    clustering_data = [vector for vector, label in data]
    clustering_data_prepared = [patch for vector in clustering_data for patch in vector]

    # TODO: this can help reduce execution time
    # indices = np.random.choice(len(clustering_data_prepared), 10000, replace=False)
    # clustering_data_sampled = np.array(clustering_data_prepared)[indices]

    # TODO: try others, maybe 500 is not best ?
    # TODO: choose between MiniBatchKMeans and the sampled method above !!!
    # k_means = KMeans(n_clusters=500, n_init=10, random_state=0)
    k_means = MiniBatchKMeans(n_clusters=500, n_init=10, batch_size=1000, random_state=0)
    k_means.fit(clustering_data_prepared)

    vocab = k_means.cluster_centers_
    return vocab


def vector_quantisation(vectors, vocab):
    words = []

    for vector in vectors:
        # Find the closest words:
        distances = np.linalg.norm(vocab - vector, axis=1)
        words.append(np.argmin(distances))

    return words


def train_classifiers(input_features, targets):
    classifiers_for_each_label = []
    classes = set(targets)
    print(classes)

    for i in range(len(classes)):
        class_targets = [1 if target == i else 0 for target in targets]
        print(f"i = {i}, {class_targets}")

        # TODO: choose parameters for this model:
        logistic_reg = LogisticRegression(max_iter=1000, random_state=42)
        logistic_reg.fit(input_features, class_targets)

        classifiers_for_each_label.append(logistic_reg)

    return classifiers_for_each_label


def make_predictions(input_features, classifiers_for_each_label):
    predictions_for_each_class = np.zeros((len(input_features), len(classifiers_for_each_label)))

    for i, classifier in enumerate(classifiers_for_each_label):
        predictions_for_each_class[:, i] = classifier.predict_proba(input_features)[:, 1]

    return np.argmax(predictions_for_each_class, axis=1)


def write_predictions_to_file(predicted, file_names, label_encoder):
    string_predictions = label_encoder.inverse_transform(predicted)
    with open("run2.txt", 'w') as output_file:
        for filename, prediction in zip(file_names, string_predictions):
            output_file.write(f"{filename} {prediction}\n")


training_data = read_images(os.path.join("..", "training"), "training")
testing_data = read_images(os.path.join("..", "testing"), "testing")
testing_filenames = [filename for image, filename in testing_data]
check_size_and_scale([image for image, label in training_data], "Training")
check_size_and_scale([image for image, filename in testing_data], "Testing")


# TODO: maybe try different patch_size and sample_frequency, other than 8 and 4 ?
training_patches = [(split_image_into_patches(image, 8, 4), label) for image, label in training_data]
testing_patches = [split_image_into_patches(image, 8, 4) for image, filename in testing_data]


training_vectors = [(flatten_patches_into_vector(patches), label) for patches, label in training_patches]
testing_vectors = [flatten_patches_into_vector(patches) for patches in testing_patches]


start = time.time()
vocabulary = learn_vocabulary(training_vectors)
end = time.time()
print(f"Clustering took {end - start} seconds.")


start = time.time()
training_quantized_words = [vector_quantisation(vector, vocabulary) for vector, label in training_vectors]
testing_quantized_words = [vector_quantisation(vector, vocabulary) for vector in testing_vectors]
end = time.time()
print(f"Quantization took {end - start} seconds.")


# Draw a histogram over the visual word counts for each image:
bins = np.arange(len(vocabulary+1))
training_feature_vectors = [np.histogram(word, bins=bins, density=True)[0] for word in training_quantized_words]
testing_feature_vectors = [np.histogram(word, bins=bins, density=True)[0] for word in testing_quantized_words]
print(len(training_feature_vectors))


labels = [label.lower() for vector, label in training_vectors]
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)


# TODO: maybe not 20% for testing ?
X_train, X_test, y_train, y_test = train_test_split(training_feature_vectors, labels_encoded, test_size=0.2,
                                                    random_state=42)
classifiers_for_precision_score = train_classifiers(X_train, y_train)
predictions_for_precision_score = make_predictions(X_test, classifiers_for_precision_score)
accuracy = accuracy_score(y_test, predictions_for_precision_score)
precision = precision_score(y_test, predictions_for_precision_score, average='macro')
print("Accuracy:", accuracy)
print("Average Precision:", precision)


start = time.time()
one_vs_all_classifiers = train_classifiers(training_feature_vectors, labels_encoded)
end = time.time()
print(f"Training classifiers took {end - start} seconds.")
final_predictions = make_predictions(testing_feature_vectors, one_vs_all_classifiers)
write_predictions_to_file(final_predictions, testing_filenames, encoder)
