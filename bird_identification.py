import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import io
import main

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# This function will handle the image and make predictions
def predict_bird(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize image to 224x224 as required by MobileNetV2
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for MobileNetV2
    
    # Predict the class
    predictions = model.predict(img_array)
    
    # Decode predictions into human-readable labels
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0][1]  # Return the predicted class name (bird name)

# Example usage
image_path = 'path_to_image.jpg'  # Replace with the actual image path
bird_name = predict_bird(image_path)
print(f"The bird in the image is: {bird_name}")
