import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import seaborn as sns

IMAGE_SIZE = (200, 200)
PCA_COMPONENTS = 0.98
NUM_SAMPLES = 3

CLASSES = {'no_tumor': 0, 'pituitary_tumor': 1, 'glioma_tumor': 2, 'meningioma_tumor': 3}

def load_and_preprocess_data(classes, image_size):
    X, Y = [], []
    for cls, label in classes.items():
        pth = f"C:\\Users\\mvy48\\OneDrive\\Desktop\\Training\\{cls}"
        for filename in os.listdir(pth):
            img = cv2.imread(os.path.join(pth, filename), 0)
            img = cv2.resize(img, image_size)
            X.append(img.flatten() / 255.0)
            Y.append(label)
    return np.array(X), np.array(Y)

X, Y = load_and_preprocess_data(CLASSES, IMAGE_SIZE)

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=10, test_size=0.2)

def apply_pca(xtrain, xtest, n_components=PCA_COMPONENTS):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca_train = pca.fit_transform(xtrain)
    pca_test = pca.transform(xtest)
    return pca, pca_train, pca_test

def train_svm(xtrain, ytrain):
    from sklearn.svm import SVC
    sv = SVC()
    sv.fit(xtrain, ytrain)
    return sv

pca, pca_train, pca_test = apply_pca(xtrain, xtest)

sv = train_svm(pca_train, ytrain)

train_predictions = sv.predict(pca_train)

train_accuracy = accuracy_score(ytrain, train_predictions)

print("Training Accuracy:", train_accuracy * 100, "%")

def display_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


display_confusion_matrix(ytrain, train_predictions, list(CLASSES.keys()))

def detect_tumor_with_similarity(image_path, dec, training_data, pca_model, sv_model, num_samples=NUM_SAMPLES):
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    
    input_img = cv2.imread(image_path, 0)
    input_img = cv2.resize(input_img, IMAGE_SIZE).flatten() / 255.0

    input_img_pca = pca_model.transform([input_img])

    prediction = sv_model.predict(input_img_pca)

    result = dec[prediction[0]]


    similarity_percentage = sv_model.decision_function(input_img_pca)[0]

    distances = np.sum((pca_train - input_img_pca) ** 2, axis=1)
    similar_indices = np.argsort(distances)[:num_samples]

    plt.figure(figsize=(15, 5))
    plt.suptitle("Brain Tumor Detection Result: {} (Similarity: {:.2%})".format(result, float(similarity_percentage[0])))

    for i, idx in enumerate(similar_indices):
        plt.subplot(1, num_samples + 1, i + 1)
        sample_img = training_data[idx].reshape(*IMAGE_SIZE)
        plt.imshow(sample_img, cmap='gray')
        plt.title(f"Sample {i + 1}")
        plt.axis('off')

    plt.subplot(1, num_samples + 1, num_samples + 1)
    plt.imshow(input_img.reshape(*IMAGE_SIZE), cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.show()
 
    return result, similarity_percentage[0]

def browse_image():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                           filetypes=(("JPEG files", "*.jpg"), ("all files", "*.*")))
    test_image_path_var.set(file_path)

def detect_tumor_from_gui():
    test_image_path = test_image_path_var.get()
    result, similarity_percentage = detect_tumor_with_similarity(test_image_path, dec, xtrain, pca, sv)
    result_label.config(text=f"Detection Result: {result}")
    similarity_label.config(text=f"Similarity Percentage: {similarity_percentage}")

    # Display confusion matrix for test data
    test_predictions = sv.predict(pca_test)
    display_confusion_matrix(ytest, test_predictions, list(CLASSES.keys()))

root = tk.Tk()
root.title("Brain Tumor Detection")

root.geometry("400x400")

test_image_path_var = tk.StringVar()

label = tk.Label(root, text="Select Test Image:")
label.pack()

browse_button = tk.Button(root, text="Browse", command=browse_image)
browse_button.pack()

test_image_path_entry = tk.Entry(root, textvariable=test_image_path_var, state="readonly")
test_image_path_entry.pack()

detect_button = tk.Button(root, text="Detect Tumor", command=detect_tumor_from_gui)
detect_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

similarity_label = tk.Label(root, text="")
similarity_label.pack()

dec = {v: k for k, v in CLASSES.items()}

root.mainloop()
