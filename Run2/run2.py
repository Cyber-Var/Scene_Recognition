import os
import time

import cv2
from statistics import mode
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def read_images(file_path, name):
    try:
        classes = os.listdir(file_path)
        images_map = []

        for label in classes:
            label_folder = os.path.join(file_path, label)
            if os.path.isdir(label_folder):
                images = os.listdir(os.path.join(file_path, label))
                for image_name in images:
                    if image_name.lower().endswith(('.jpg', '.jpeg')):
                        image_file = os.path.join(label_folder, image_name)
                        images_map.append((cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), label))

        return images_map
    except Exception as e:
        print(f'Error when reading {name} data', e)


def check_size_and_scale(data):
    null = 0
    heights = []
    widths = []
    maxs = []
    mins = []
    scales = []
    for image, label in data:
        if image is not None:
            heights.append(image.shape[0])
            widths.append(image.shape[1])
            mins.append(image.min())
            maxs.append(image.max())
            scales.append(image.max() - image.min())
        else:
            null += 1

    print("Number of images of type None =", null)
    print(f"Height: mode = {mode(heights)}, min = {min(heights)}, max = {max(heights)}")
    print(f"Width: mode = {mode(widths)}, min = {min(widths)}, max = {max(widths)}")
    print(f"Scale: average min = {mode(mins)}, average max = {mode(maxs)}, average scale range = {mode(scales) + 1}")


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




training_data = read_images(os.path.join("..", "training"), "training")
check_size_and_scale(training_data)

print(training_data[1499][0].shape)

# TODO: maybe try different patch_size and sample_frequency, other than 8 and 4 ?
training_patches = [(split_image_into_patches(image, 8, 4), label) for image, label in training_data]
print(training_patches[1499][0].shape)

training_vectors = [(flatten_patches_into_vector(patches), label) for patches, label in training_patches]
print(training_vectors[1499][0].shape)

start = time.time()
vocabulary = learn_vocabulary(training_vectors)
end = time.time()
print(f"Clustering took {end - start} seconds.")
# print(vocabulary)

start = time.time()
quantized_words = [vector_quantisation(vector, vocabulary) for vector, label in training_vectors]
end = time.time()
print(f"Quantization took {end - start} seconds.")
# print(quantized_words)

start = time.time()
# Draw a histogram over the visual word counts for each image:
bins = np.arange(len(vocabulary+1))
feature_vectors = [np.histogram(word, bins=bins, density=True)[0] for word in quantized_words]
end = time.time()
print(f"Histograms took {end - start} seconds.")
# print(feature_vectors)



# test_image = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#                        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#                        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#                        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#                        [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7],
#                        [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7],
#                        [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7],
#                        [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7],
#                        [8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11],
#                        [8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11],
#                        [8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11],
#                        [8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11],
#                        [12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15],
#                        [12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15],
#                        [12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15],
#                        [12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15]])
