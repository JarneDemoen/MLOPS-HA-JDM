import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization,concatenate
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras import backend as K

def getTargets(filepaths: List[str]) -> List[str]:
    targets = []
    for fp in filepaths:
        fp_splitted = fp.split('/')
        label = fp_splitted[5]
        if label == 'ds_lung_masks':
            image = cv2.imread(fp)
            targets.append(image)
    return np.array(targets)

def getFeatures(filepaths: List[str]) -> np.array:
    features = []
    for fp in filepaths:
        fp_splitted = fp.split('/')
        label = fp_splitted[5]
        if label == 'ds_lung_images':
            image = cv2.imread(fp)
            features.append(image)
    return np.array(features)

def buildModel(inputShape: tuple) -> Sequential:
    #Encoder
    print("InputShape",inputShape)
    input_img = Input(shape=inputShape)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoder = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoder)


    return autoencoder