import os
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image

LUNGS = ['ds_lung_images', 'ds_lung_masks']

def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_path = os.path.join(os.environ.get('AZUREML_MODEL_DIR'), 'lung-cnn')

    model = load_model(model_path)

def run(image):
    data = json.loads(image)
    img = np.asarray(data['data'])
    print(img.shape)
    images_to_predict = np.expand_dims(img, axis=0)
    predictions = model.predict(images_to_predict)
    # classifications = predictions.argmax(axis=1)

    return predictions