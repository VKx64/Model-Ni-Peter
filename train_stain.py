# ===========================
# Import Necessary Libraries
# ===========================
import os
import logging
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ==========================================
# Configuration Parameters for Image Model
# ==========================================
IMAGE_SIZE = (100, 100)
DATASET_DIR = "dataset/stain"  # Single folder for both train and test images
IMAGE_MODEL_SAVE_PATH = "models/stain_model.keras"
TFLITE_MODEL_SAVE_PATH = "models/stain_model.tflite"  # Path to save the TFLite model
IMAGE_GRAPH_SAVE_PATH = "graphs/"
CLASS_NAMES = []  # To be dynamically generated
TEST_SIZE = 0.2  # Set the proportion of data to be used as test data

# Ensure the model directory and graph save path exist
os.makedirs(os.path.dirname(IMAGE_MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(IMAGE_GRAPH_SAVE_PATH, exist_ok=True)

# ================================
# Generate Dynamic CLASS_NAMES from Directory
# ================================
def generate_class_names(directory):
    """
    Generate class names based on the folder names in the dataset directory.
    """
    class_names = [folder_name for folder_name in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder_name))]
    class_names.sort()  # Sort alphabetically
    return class_names

# Dynamically generate CLASS_NAMES based on the folder names in the dataset directory
CLASS_NAMES = generate_class_names(DATASET_DIR)
logging.info(f"Class names dynamically generated: {CLASS_NAMES}")

# ================================
# Helper Functions for Image Model
# ================================

def plot_image_training_history(history, save_path):
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Image Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Image Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    plt.close()

def plot_image_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Image Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def save_image_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as file:
        file.write(report)

def data_dictionary(base_dir=DATASET_DIR, class_names=CLASS_NAMES):
    data_dict = {"image_path": [], "target": []}

    for idx, disease in enumerate(class_names):
        image_dir = os.path.join(base_dir, disease)

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory {image_dir} does not exist")

        for img_file in os.listdir(image_dir):
            data_dict["image_path"].append(os.path.join(image_dir, img_file))
            data_dict["target"].append(idx)

    return pd.DataFrame(data_dict)

def load_image_data(image_size=IMAGE_SIZE, test_size=TEST_SIZE):
    df = data_dictionary()

    x_data = []
    y_data = []
    
    # Read and resize images with error handling
    for i, path in enumerate(df['image_path']):
        img = cv2.imread(path)
        if img is None:
            logging.error(f"Error reading image at path: {path}. Skipping this file.")
            continue  # Skip this image if it can't be read
        
        try:
            img_resized = cv2.resize(img, image_size)
            x_data.append(img_resized)
            y_data.append(df['target'][i])
        except Exception as e:
            logging.error(f"Error resizing image at path: {path}. Skipping this file. Error: {str(e)}")
    
    # Convert lists to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Check if there's enough data to proceed
    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError("No valid images found after reading the dataset.")

    # Split into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, stratify=y_data, shuffle=True)
    
    return x_train, x_test, y_train, y_test

def build_image_model(input_shape):
    inputs = Input(shape=(input_shape,))
    
    x = Dense(1024, activation=None, kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation=None, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation=None, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation=None, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    outputs = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def extract_image_features(vgg_model, x_data):
    x_data = preprocess_input(x_data)
    features = vgg_model.predict(x_data)
    features = features.reshape(features.shape[0], -1)
    return features

# ================================
# Function to Convert Keras Model to TFLite
# ================================
def convert_to_tflite(keras_model, save_path):
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model successfully converted to TFLite and saved at {save_path}")

# ================================
# Train and Save the Image Model
# ================================
def train_image_model():
    print("Loading image data...")
    x_train, x_test, y_train, y_test = load_image_data()
    print("Image data loaded successfully.")

    print("Loading VGG19 model for feature extraction...")
    vgg19 = VGG19(include_top=False, weights='imagenet')

    print("Extracting features for training data...")
    features_train = extract_image_features(vgg19, x_train)
    print("Training features extracted.")

    print("Extracting features for testing data...")
    features_test = extract_image_features(vgg19, x_test)
    print("Testing features extracted.")

    print("Building the image model...")
    model = build_image_model(features_train.shape[1])
    print("Image model built successfully.")

    print("Starting image model training...")
    history = model.fit(
        features_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(features_test, y_test)
    )

    print("Image model training completed successfully.")

    test_loss, test_accuracy = model.evaluate(features_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    plot_image_training_history(history, IMAGE_GRAPH_SAVE_PATH)

    y_pred = model.predict(features_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    plot_image_confusion_matrix(y_test, y_pred_labels, CLASS_NAMES, IMAGE_GRAPH_SAVE_PATH)
    save_image_classification_report(y_test, y_pred_labels, CLASS_NAMES, IMAGE_GRAPH_SAVE_PATH)

    print("Saving the trained image model (Keras format)...")
    model.save(IMAGE_MODEL_SAVE_PATH)
    print("Image model saved successfully in Keras format.")

    # Convert and save the model as a TensorFlow Lite model
    print("Converting the model to TensorFlow Lite format...")
    convert_to_tflite(model, TFLITE_MODEL_SAVE_PATH)

    return model

# ===========================
# Main Function to Train the Image Model
# ===========================
if __name__ == "__main__":
    train_image_model()
