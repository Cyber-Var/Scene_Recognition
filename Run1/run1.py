from sklearn.neighbors import KNeighborsClassifier
import os 
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from PIL import Image
import numpy as np

rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)

def zero_mean_normalize(image):
    """
        Function that receives the image and applies zero mean normalization and scaling the pixels to achieve a unit standard deviation
        :param image: image
        :return the normalized image
    """
    mean_value = np.mean(image)
    std_value = np.std(image)
    normalized_image = (image - mean_value) /std_value
    return normalized_image


def resize_img(image):
    """
        Function that resizes the image into 16x16
        :param image: image
        :return the resized image
    """
    resized_img = image.resize((16, 16))
    return resized_img


def preprocess_image(image_path):
    """
        Function that opens the image path, reads the image and applies the zero_mean_normalization and resizes it. 
        :param image_path: path of the image
        :returns the preprocessed image
    """
    with Image.open(image_path) as img:

        img_sized = resize_img(img)
        tiny_image = zero_mean_normalize(img_sized)
        return tiny_image

        
def load_dataset(root_folder):
    """
        Function that opens the training folder and gets all the paths of the images and their labels
        :param root_folder: folder of the training file
        :returns the image paths and the labels of each image
    """
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




#Get our directory
script_dir = os.path.dirname(os.path.realpath(__file__))

#Get the training directory were our training dataset is
dataset_path = os.path.join(script_dir, "..", "training")
#Links of the images and the labels of each image
links, labels = load_dataset(dataset_path)

#All the image's features added to this array.
feature_vectors = []

#Preprcoess the each image 
#Preprocessing includes resizing and normalisation
for i in links:
    feature_vectors.append(preprocess_image(i).flatten())

#feature vectors of the training images
X = feature_vectors
#Labels of the imahges
y = labels

#Create a K neighbors classifier using the best n parameter which was tested and found to be 7. 
knn_classifier = KNeighborsClassifier(n_neighbors=7)
#Fit the training and its labels into the model
knn_classifier.fit(X, y)


#Outputing Test Data Predictions

#The directory where the testing images are
test_dir = os.path.join(script_dir, "..", "testing")

#Test image name
test_file_names = []
#Predicted class for each image
test_class = []

#Go through each test image pre-process them then classify it as one of the classifiers.
#Append the name of the image being classified to the text_files_names
#Append the classification of the image to the test_class
for i in range(0,2988):
    try:
        file_path_temp = os.path.join(test_dir, f"{i}.jpg")

        preprocessed_image = preprocess_image(file_path_temp).flatten()

        preprocessed_image = preprocessed_image.reshape(1, -1)

        predicted_class = knn_classifier.predict(preprocessed_image)

        predicted_class = str(predicted_class[0].lower())  # Convert numpy array to string    

        test_class.append(predicted_class)
        test_file_names.append(str(i) + ".jpg")
    except Exception as e:
        print(f"Error processing image {file_path_temp}: {e}")


#Create the run1.txt file and write all the test classification 
with open("run1.txt", 'w') as output_file:
    for m in range(0, len(test_class)):
        output_file.write(str(test_file_names[m]) + " " + str(test_class[m]) + "\n")


