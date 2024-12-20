import pandas as pd

# Load the CSV file
df = pd.read_csv('images.csv')

import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Load the pre-trained ResNet50 model without the top layer (for feature extraction)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to load and preprocess an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for ResNet50
    return img_array

# Function to process all images in a folder and its subfolders
def process_images_in_folder(base_folder):
    features_list = []

    # Walk through the base folder and its subfolders
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
                img_path = os.path.join(root, file)  # Get the full path to the image
                print(f"Processing image: {img_path}")  # Optional: Print the image path being processed

                # Preprocess the image and extract features
                img_array = preprocess_image(img_path)
                features = model.predict(img_array)  # Extract features
                features_list.append((features.flatten(), img_path))  # Store the features and path

    return features_list

# Example usage
base_folder = 'E:/Re-PolyVore'  # Path to your main folder (adjust to your directory)
features_list = process_images_in_folder(base_folder)

# Print the shape of the feature vector for the first image as an example
if features_list:
    print("Feature vector shape for first image:", features_list[0][0].shape)
