# ===========================
# Import Necessary Libraries
# ===========================
import os
import numpy as np
import cv2
import tensorflow as tf
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from flask import Flask, request, jsonify

# ===========================
# Flask App Initialization
# ===========================
app = Flask(__name__)

# ===========================
# Configuration Parameters
# ===========================
IMAGE_SIZE = (100, 100)
CLASS_NAMES_STAIN = []  # Class names for the stain model
CLASS_NAMES_FABRIC = []  # Class names for the fabric model
MODEL_PATH_STAIN = "models/stain_model.keras"  # Path to the stain model
MODEL_PATH_FABRIC = "models/fabric_model.keras"  # Path to the fabric model

# ===========================
# Load Class Names
# ===========================
def load_class_names(directory):
    """
    Load class names from the dataset directory used for training.
    """
    class_names = [folder_name for folder_name in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder_name))]
    class_names.sort()  # Sort alphabetically
    return class_names

# Load the class names dynamically
CLASS_NAMES_STAIN = load_class_names("dataset/stain")
CLASS_NAMES_FABRIC = load_class_names("dataset/fabric")

# ===========================
# Load and Preprocess Image
# ===========================
def load_and_preprocess_image(image_data, image_size=IMAGE_SIZE):
    """
    Preprocess an image for prediction.
    """
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not process the image data.")

    img_resized = cv2.resize(img, image_size)
    img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))  # Add batch dimension
    return img_preprocessed

# ===========================
# Extract Features from Image
# ===========================
def extract_image_features(vgg_model, image_data):
    """
    Extract features from the image using a pre-trained VGG19 model.
    """
    image_data = preprocess_input(image_data)
    features = vgg_model.predict(image_data)
    features = features.reshape(features.shape[0], -1)  # Flatten the features
    return features

# ===========================
# Perform Prediction
# ===========================
def predict_image(image_data, model_path, class_names):
    """
    Load the trained model, preprocess the image, extract features, and return confidence scores for all classes.
    """
    # Load the trained model
    model = load_model(model_path)

    # Load VGG19 model for feature extraction
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

    # Extract features from the image
    image_features = extract_image_features(vgg19, image_data)

    # Perform prediction
    predictions = model.predict(image_features)

    # Output confidence scores for all classes
    confidence_scores = predictions[0]  # Extract the first (and only) set of predictions

    response = {}
    for i, class_name in enumerate(class_names):
        response[class_name] = round(float(confidence_scores[i] * 100), 2)

    return response

# ===========================
# Decode Base64 Image
# ===========================
def decode_base64_image(base64_str):
    """
    Decode a base64 encoded image string.
    """
    try:
        image_data = base64.b64decode(base64_str)
        return image_data
    except Exception as e:
        raise ValueError(f"Failed to decode base64 string: {str(e)}")

# ===========================
# New Endpoint to Predict Using Both Models
# ===========================
@app.route('/predict_both', methods=['POST'])
def predict_both():
    try:
        # Check if an image file is provided
        if 'image' in request.files:
            image_file = request.files['image'].read()
            if not image_file:
                return jsonify({'error': 'Image file is empty'}), 400

            image_data = load_and_preprocess_image(image_file)

        # Check if a base64 image is provided
        elif 'image_base64' in request.json:
            base64_str = request.json['image_base64']
            image_file = decode_base64_image(base64_str)
            image_data = load_and_preprocess_image(image_file)

        else:
            return jsonify({'error': 'No image provided'}), 400

        # Perform predictions for both models
        combined_response = {}
        combined_response['stain_model'] = predict_image(image_data, MODEL_PATH_STAIN, CLASS_NAMES_STAIN)
        combined_response['fabric_model'] = predict_image(image_data, MODEL_PATH_FABRIC, CLASS_NAMES_FABRIC)

        return jsonify({'predictions': combined_response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===========================
# Run Flask App
# ===========================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
