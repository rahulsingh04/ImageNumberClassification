import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Configure TensorFlow session
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Notebooks/MobileNet.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# Define the function for model prediction
def model_predict(img_path, model):
    print(img_path)
    img = load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    x = x / 255.0  # Scaling
    x = np.expand_dims(x, axis=0)
    return model.predict(x)

# Create an 'uploads' directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the route for the home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Define the route for image upload and prediction
@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        preds = model_predict(file_path, model)
        result1 = preds[0]
        result=np.argmax(preds, axis=1)[0]
        print("preds Is :->", result)

        return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
