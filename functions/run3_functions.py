import os
import numpy as np
from PIL import Image


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


def get_image_size(image_path):
    # Open the image and get its size
    with Image.open(image_path) as img:
        return img.size


def calculate_images_int_average_size(image_paths):
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
    # Resize the image, resize() takes (width, height) as input
    resized_img = image.resize((average_width, average_height))
    return resized_img


def process_image_densenet_3d(image_path, average_height, average_width):
    with Image.open(image_path) as img:
        resized_image = resize_image(img, average_height, average_width)
        image_np_array = np.array(resized_image)
        # convert the image to 3 channels, so it can be processed by the model
        return np.stack((image_np_array,) * 3, axis=-1)  # Convert single-channel to three identical channels


def zero_mean_normalize(image):
    mean_value = np.mean(image)
    std_value = np.std(image)
    normalized_image = (image - mean_value) /std_value
    return normalized_image


def process_image_alexnet(image_path, average_height, average_width):
    with Image.open(image_path) as img:
        resized_image = resize_image(img, average_height, average_width)
        image_np_array = np.array(resized_image)
        nor_img = zero_mean_normalize(image_np_array)
        return nor_img