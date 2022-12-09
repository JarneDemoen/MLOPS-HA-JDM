import io
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from skimage.io import imread, imshow
from skimage import transform
import os
from fastapi.responses import StreamingResponse, FileResponse, Response
from starlette.responses import FileResponse
import matplotlib.pyplot as plt

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LUNGS = ['ds_lung_images', 'ds_lung_masks']

model = load_model('outputs/lungs-cnn', compile=False) # Model_name here!

@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):

    # convert img to numpy array
    img = Image.open(img.file)
    print('img',img)
    img = np.array(img)
    print('img',img)
    lung_images = []
    img = transform.resize(img, (128, 128,3), mode='constant', anti_aliasing=True)
    print('img',img)
    lung_images.append(img)
    print('lung_images',lung_images)

    lung_images = np.array(lung_images)
    print('lung_images',lung_images)
    lung_images = lung_images.reshape(len(lung_images), 128, 128, 3)
    print('lung_images',lung_images)

    print('lung_images',lung_images.shape)

    # predict
    output = model.predict(lung_images)
    output = np.squeeze(output)
    fig = plt.figure()
    plt.imshow(output)
    plt.savefig('prediction.jpg')
    # output = Image.fromarray(output, mode='RGB')
    # output.save('prediction.jpg')    
    # return Response(content=io.BytesIO(output.tobytes()), media_type="image/png")
    # return StreamingResponse(io.BytesIO(output.tobytes()), media_type="image/png")
    

    return FileResponse(media_type='application/octet-stream',filename='prediction.jpg',path='prediction.jpg')
    # return FileResponse('prediction.jpg')