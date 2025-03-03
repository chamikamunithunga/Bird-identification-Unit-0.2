from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import requests

app = Flask(__name__)

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Set up file upload directory
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the predict_bird function
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
    bird_name = decoded_predictions[0][1]  # Return the predicted class name (bird name)
    return bird_name

# Function to query the eBird API for information about the bird
# Function to query the eBird API for information about the bird
def get_bird_info(bird_name):
    API_KEY = os.getenv('omcelrsi7rt2')  # Retrieve the API key from the environment
    
    url = f"https://api.ebird.org/v2/data/obs/geo/recent?species={bird_name}"
    
    headers = {
        'X-eBirdApiToken': API_KEY
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        bird_data = response.json()
        if not bird_data:
            return {"message": "No additional bird data found."}
        return bird_data
    else:
        return {"message": "Error fetching bird data from eBird API."}



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict bird
        bird_name = predict_bird(filepath)
        
        # Fetch additional information using the eBird API
        bird_info = get_bird_info(bird_name)
        
        if bird_info:
            return f"The bird in the image is: {bird_name}. Here's some information from eBird: {bird_info}"
        else:
            return f"The bird in the image is: {bird_name}. However, no additional information was found from eBird."
    else:
        return 'Invalid file format. Please upload a PNG or JPG image.'

if __name__ == '__main__':
    app.run(debug=True)
