import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Download and Extract the Dataset
dataset_path = kagglehub.dataset_download("crawford/emnist")

# Step 2: Load the Dataset
train_csv = f"{dataset_path}/emnist-balanced-train.csv"
test_csv = f"{dataset_path}/emnist-balanced-test.csv"

# Load CSV files
train_data = pd.read_csv(train_csv, header=None)
test_data = pd.read_csv(test_csv, header=None)

# Separate labels and images
train_labels = train_data.iloc[:, 0].values  # First column is labels
train_images = train_data.iloc[:, 1:].values  # Remaining columns are pixel values

test_labels = test_data.iloc[:, 0].values
test_images = test_data.iloc[:, 1:].values

# Step 3: Normalize & Reshape Images
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Step 4: Split into Train/Validation Sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Function to Show Sample Images
def show_sample_images():
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(train_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {train_labels[i]}")
        plt.axis('off')
    plt.show()

# Save preprocessed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("test_images.npy", test_images)
np.save("test_labels.npy", test_labels)

print("Dataset Loaded & Preprocessed Successfully!")
