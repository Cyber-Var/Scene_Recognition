from PIL import Image
import numpy as np
import os

def zero_mean_normalize(image):
    mean_value = np.mean(image)
    std_value = np.std(image)
    normalized_image = (image - mean_value) /std_value
    return normalized_image


def resize_img(image):
    resized_img = image.resize((16, 16))
    return resized_img

def preprocess_image(image_path):
    with Image.open(image_path) as img:
    
        img_sized = resize_img(img)
        tiny_image = zero_mean_normalize(img_sized)
        return tiny_image

        
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

    return image_paths,labels


