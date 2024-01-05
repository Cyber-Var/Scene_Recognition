from sklearn.neighbors import KNeighborsClassifier
import os 
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)


from functions.functions import *



script_dir = os.path.dirname(os.path.realpath(__file__))

dataset_path = os.path.join(script_dir, "..", "training")
links, labels = load_dataset(dataset_path)

feature_vectors = []

for i in links:
    feature_vectors.append(preprocess_image(i).flatten())

X = feature_vectors
y = labels


knn_classifier = KNeighborsClassifier(n_neighbors=11)
knn_classifier.fit(X, y)

#Outputing Test Data Predictions

test_dir = os.path.join(script_dir, "..", "testing")

test_file_names = []
test_class = []
for i in range(0,2988):
    try:
        file_path_temp = os.path.join(test_dir, f"{i}.jpg")

        preprocessed_image = preprocess_image(file_path_temp).flatten()

        preprocessed_image = preprocessed_image.reshape(1, -1)

        predicted_class = knn_classifier.predict(preprocessed_image)

        test_class.append(predicted_class)
        test_file_names.append(str(i) + ".jpg")
    except Exception as e:
        print(f"Error processing image {file_path_temp}: {e}")



with open("run1.txt", 'w') as output_file:
    for m in range(0, len(test_class)):
        output_file.write(str(test_file_names[m]) + " " + str(test_class[m][0]) + "\n")











# # TESTING ON TRAIN DATA


# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# # Initialize k-NN classifier
# optimal_k = 11  # You can choose an optimal k based on your experiments
# knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)

# # Train the classifier on the training set
# knn_classifier.fit(X_train, y_train)


# # Make predictions on the testing set
# y_pred = knn_classifier.predict(X_test)

# # Evaluate the model's accuracy on the testing set
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy on the testing set: {accuracy}")





