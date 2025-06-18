import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import cv2
import os

SIZE = 128
DIM = (SIZE, SIZE)

def predict_deepfake(image_path, model_path='binary-pred.h5'):
    try:
        model = load_model(model_path, compile=False)

        img = cv2.imread(image_path)
        if img is None:
            return f"Error: Could not read image from {image_path}"

        img_resized = cv2.resize(img, DIM)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_preprocessed = preprocess_input(img_rgb)
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        prediction = model.predict(img_batch)
        probability_deepfake = prediction[0][0]

        if probability_deepfake > 0.5:
            confidence = probability_deepfake
            result = "Deepfake"
        else:
            confidence = 1 - probability_deepfake
            result = "Real"

        return f"Prediction: {result} (Confidence: {confidence:.4f})"

    except FileNotFoundError as e:
        return f"Error loading model: {e}. Make sure '{model_path}' is in the correct directory."
    except Exception as e:
        return f"An error occurred during prediction: {e}"

if _name_ == "_main_":
    image_to_predict = r"C:/Users/yahdr/Documents/ML(project)/image(1).webp"
    model_file = 'binary-pred.h5'

    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found. Please ensure it's in the same directory or provide the full path.")
    elif not os.path.exists(image_to_predict):
         print(f"Error: Image file '{image_to_predict}' not found. Please provide the correct path.")
    else:
        result = predict_deepfake(image_to_predict, model_file)
        print(result)