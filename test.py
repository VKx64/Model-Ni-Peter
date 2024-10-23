# ===========================
# Import Necessary Libraries
# ===========================
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import VGG19

# ==========================================
# Configuration Parameters for Prediction
# ==========================================
IMAGE_SIZE = (100, 100)  # Must match the size used during training
DATASET_DIR = "dataset/fabric"  # Directory containing the image class folders
KERAS_MODEL_PATH = "models/skin_model.keras"  # Path to the Keras model
TFLITE_MODEL_PATH = "models/skin_model.tflite"  # Path to the TFLite model

# Generate class names dynamically from the dataset directory (same as in training)
def generate_class_names(directory):
    """
    Generate class names based on the folder names in the dataset directory.
    """
    class_names = [folder_name for folder_name in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder_name))]
    class_names.sort()  # Sort alphabetically
    return class_names

# Dynamically generate CLASS_NAMES based on the folder names in the dataset directory
CLASS_NAMES = generate_class_names(DATASET_DIR)

# ==========================================
# Function to Preprocess the Image
# ==========================================
def preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Preprocess an image for prediction.
    - Load the image.
    - Resize it to the required input size.
    - Preprocess it using VGG19 preprocessing.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_resized = cv2.resize(img, image_size)
    img_preprocessed = preprocess_input(img_resized)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension
    return img_expanded

# ==========================================
# Function to Predict with Keras Model
# ==========================================
def predict_with_keras(image_path):
    """
    Load the saved Keras model and make predictions on the provided image.
    """
    # Load the Keras model
    model = load_model(KERAS_MODEL_PATH)

    # Preprocess the image
    image = preprocess_image(image_path)

    # Extract features using VGG19 (matching the training process)
    vgg19 = VGG19(include_top=False, weights='imagenet')
    image_features = vgg19.predict(image)
    image_features = image_features.reshape(1, -1)  # Flatten the features

    # Predict the class
    prediction = model.predict(image_features)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Return the folder name (class name)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    print(f"Predicted class (Keras): {predicted_class_name}")
    return predicted_class_name

# ==========================================
# Function to Predict with TFLite Model
# ==========================================
def predict_with_tflite(image_path):
    """
    Load the saved TFLite model and make predictions on the provided image.
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    image = preprocess_image(image_path)

    # Extract features using VGG19 (matching the training process)
    vgg19 = VGG19(include_top=False, weights='imagenet')
    image_features = vgg19.predict(image)
    image_features = image_features.reshape(1, -1)  # Flatten the features

    # Set the input tensor for the TFLite model
    interpreter.set_tensor(input_details[0]['index'], image_features)

    # Run inference
    interpreter.invoke()

    # Get the prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data, axis=1)[0]

    # Return the folder name (class name)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    print(f"Predicted class (TFLite): {predicted_class_name}")
    return predicted_class_name

# ==========================================
# Main Prediction Function
# ==========================================
if __name__ == "__main__":
    # Specify the path to the image you want to predict
    IMAGE_PATH = "test_carduroy.png"  # Replace with the path to your image

    # Predict with Keras model
    predict_with_keras(IMAGE_PATH)

    # Predict with TFLite model
    predict_with_tflite(IMAGE_PATH)
