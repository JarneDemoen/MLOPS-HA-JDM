import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from skimage.color import rgb2gray
from skimage.io import imread, imshow
from skimage import data, color, io, filters, morphology,transform, exposure, feature, util
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LUNGS = ['ds_lung_images', 'ds_lung_masks']

model = load_model('outputs/lungs-cnn') # Model_name here!

@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    img = Image.open(img.file)
    # original_image = original_image.resize((128, 128))
    img = rgb2gray(img)
    img = transform.resize(img, (128, 128,3), mode='constant', anti_aliasing=True)
    img = np.array(img)
    
    mask = model.predict(img)

    return mask