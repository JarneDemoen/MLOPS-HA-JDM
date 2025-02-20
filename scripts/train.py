import argparse
import os
from glob import glob
import random
import numpy as np

# This time we will need our Tensorflow Keras libraries, as we will be working with the AI training now
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# This AzureML package will allow to log our metrics etc.
from azureml.core import Run

# Important to load in the utils as well!
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--training-folder', type=str, dest='training_folder', help='training folder mounting point.')
parser.add_argument('--testing-folder', type=str, dest='testing_folder', help='testing folder mounting point.')
parser.add_argument('--max-epochs', type=int, dest='max_epochs', help='The maximum epochs to train.')
parser.add_argument('--seed', type=int, dest='seed', help='The random seed to use.')
parser.add_argument('--initial-learning-rate', type=float, dest='initial_lr', help='The initial learning rate to use.')
parser.add_argument('--batch-size', type=int, dest='batch_size', help='The batch size to use during training.')
parser.add_argument('--patience', type=int, dest='patience', help='The patience for the Early Stopping.')
parser.add_argument('--model-name', type=str, dest='model_name', help='The name of the model to use.')
args = parser.parse_args()


training_folder = args.training_folder
print('Training folder:', training_folder)

testing_folder = args.testing_folder
print('Testing folder:', testing_folder)

MAX_EPOCHS = args.max_epochs # Int
INITIAL_LEARNING_RATE = args.initial_lr # Float
BATCH_SIZE = args.batch_size # Int
PATIENCE = args.patience # Int
MODEL_NAME = args.model_name # String


# As we're mounting the training_folder and testing_folder onto the `/mnt/data` directories, we can load in the images by using glob.
training_paths = glob(os.path.join('./data/train', '**', 'processed_lungs', '**', '*.png'), recursive=True)
testing_paths = glob(os.path.join('./data/test', '**', 'processed_lungs', '**', '*.png'), recursive=True)

print("Training samples:", len(training_paths))
print("Testing samples:", len(testing_paths))

# Parse to Features and Targets for both Training and Testing. Refer to the Utils package for more information
X_train = getFeatures(training_paths)
y_train = getTargets(training_paths)

X_test = getFeatures(testing_paths)
y_test = getTargets(testing_paths)

X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)

y_train = tf.cast(y_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)

print('Shapes:')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create an output directory where our AI model will be saved to.
# Everything inside the `outputs` directory will be logged and kept aside for later usage.
model_path = os.path.join('outputs', MODEL_NAME)
os.makedirs(model_path, exist_ok=True)

## START OUR RUN context.
## We can now log interesting information to Azure, by using these methods.
run = Run.get_context()

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2.*intersection + smooth)/(K.sum(K.square(y_true),-1)+ K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# Save the best model, not the last
cb_save_best_model = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                         monitor='val_loss',
                                                         save_best_only=True,
                                                         verbose=1)

# Early stop when the val_los isn't improving for PATIENCE epochs
cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_dice_coef', 
                                              patience= PATIENCE,
                                              verbose=1,
                                              restore_best_weights=True)

# Reduce the Learning Rate when not learning more for 4 epochs.
cb_reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(factor=.5, patience=4, verbose=1)

opt = SGD(lr=INITIAL_LEARNING_RATE, decay=INITIAL_LEARNING_RATE / MAX_EPOCHS) # Define the Optimizer

autoencoder = buildModel((128, 128, 3))
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
autoencoder.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])

# Add callback LogToAzure class to log to AzureML
class LogToAzure(keras.callbacks.Callback):
    '''Keras Callback for realtime logging to Azure'''
    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        # Log all log data to Azure
        for k, v in logs.items():
            self.run.log(k, v)

# train the network
autoencoder.fit(X_train, y_train, epochs=50, batch_size=8, shuffle=True, callbacks=[LogToAzure(run), cb_save_best_model, cb_early_stop], validation_data=(X_test, y_test), verbose=1)

print("[INFO] evaluating network...")

print("DONE TRAINING. AI model has been saved to the outputs.")
