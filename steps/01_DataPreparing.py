from ctypes import resize
from glob import glob
import json
import os
from datetime import datetime
import math
import random
import shutil

from utils import connectWithAzure

import cv2
from dotenv import load_dotenv
from azureml.core import Dataset
from azureml.data.datapath import DataPath



# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()
# LUNG_IMAGES = os.environ.get('LUNG_IMAGES')
# LUNG_MASKS = os.environ.get('LUNG_MASKS')
# LUNGS = [LUNG_IMAGES, LUNG_MASKS]
LUNGS = os.environ.get('LUNGS').split(',') # When using Github Actions
SEED = int(os.environ.get('RANDOM_SEED'))
TRAIN_TEST_SPLIT_FACTOR = float(os.environ.get('TRAIN_TEST_SPLIT_FACTOR'))

def processAndUploadLungImages(datasets, data_path, processed_path, ws, dataset_name):

    # We can't use mount on these machines, so we'll have to download them

    lung_dataset_path = os.path.join(data_path, 'lungs', dataset_name)
    print(f'Creating directory for {dataset_name} images at {lung_dataset_path} ...')
    # Get the dataset name for this lung_dataset, then download to the directory
    datasets[dataset_name].download(lung_dataset_path, overwrite=True) # Overwriting means we don't have to delete if they already exist, in case something goes wrong.
    print('Downloading all the images')

    # Get all the image paths with the `glob()` method.
    print(f'Resizing all images for {dataset_name} ...')
    # print(os.listdir(lung_dataset_path))
    # print(f"lung dataset path: {lung_dataset_path}")
    # print(f"dataset name: {dataset_name}")
    image_paths = glob(f"{lung_dataset_path}/*.png")
    # image_paths = glob(f'{lung_dataset_path}/{dataset_name}/**/*.png') # CHANGE THIS LINE IF YOU NEED TO GET YOUR dataset_nameS IN THERE IF NEEDED!
    # print("image paths: ", image_paths)
    # Process all the images with OpenCV. Reading them, then resizing them to 64x64 and saving them once more.
    print(f"Processing {len(image_paths)} images")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128)) # Resize to a square of 128, 128  
        cv2.imwrite(os.path.join(processed_path, dataset_name, image_path.split('/')[-1]), image)
    print(f'... done resizing. Stopping context now...')
    
    # Upload the directory as a new dataset
    print(f'Uploading directory now ...')
    resized_dataset = Dataset.File.upload_directory(
                        # Enter the sourece directory on our machine where the resized pictures are
                        src_dir = os.path.join(processed_path, dataset_name),
                        # Create a DataPath reference where to store our images to. We'll use the default datastore for our workspace.
                        target = DataPath(datastore=ws.get_default_datastore(), path_on_datastore=f'processed_lungs/{dataset_name}'),
                        overwrite=True)

    print('... uploaded images, now creating a dataset ...')

    # Make sure to register the dataset whenever everything is uploaded.
    new_dataset = resized_dataset.register(ws,
                            name=f'resized_{dataset_name}',
                            description=f'{dataset_name} images resized tot 128, 128',
                            tags={'lungs': dataset_name, 'AI-Model': 'CNN', 'GIT-SHA': os.environ.get('GIT_SHA')}, # Optional tags, can always be interesting to keep track of these!
                            create_new_version=True)
    print(f" ... Dataset id {new_dataset.id} | Dataset version {new_dataset.version}")
    print(f'... Done. Now freeing the space by deleting all the images, both original and processed.')
    emptyDirectory(lung_dataset_path)
    print(f'... done with the original images ...')
    emptyDirectory(os.path.join(processed_path, dataset_name))
    print(f'... done with the processed images. On to the next dataset, if there are still!')

def emptyDirectory(directory_path):
    shutil.rmtree(directory_path)

def prepareDataset(ws):
    data_folder = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_folder, exist_ok=True)
    print("LUNGS: ", LUNGS)
    for dataset_name in LUNGS:
        print(f"Creating directory for {dataset_name} images at {data_folder} ...")
        os.makedirs(os.path.join(data_folder, 'lungs', dataset_name), exist_ok=True)

    # Define a path to store the lung images onto. We'll choose for `data/processed/lungs` this time. Again, create subdirectories for all the lungs
    processed_path = os.path.join(os.getcwd(), 'data', 'processed_lungs', 'lungs')
    os.makedirs(processed_path, exist_ok=True)
    for dataset_name in LUNGS:
        os.makedirs(os.path.join(processed_path, dataset_name), exist_ok=True)

    datasets = Dataset.get_all(workspace=ws) # Make sure to give our workspace with it
    for dataset_name in LUNGS:
        processAndUploadLungImages(datasets, data_folder, processed_path, ws, dataset_name)

def trainTestSplitData(ws):

    training_datapaths = []
    testing_datapaths = []
    default_datastore = ws.get_default_datastore()
    for dataset_name in LUNGS:
        # Get the dataset by name
        lung_dataset = Dataset.get_by_name(ws, f"resized_{dataset_name}")
        print(f'Starting to process {dataset_name} images.')

        # Get only the .JPG images
        lung_images = [img for img in lung_dataset.to_path() if img.split('.')[-1] == 'jpg']

        print(f'... there are about {len(lung_images)} images to process.')

        ## Concatenate the names for the dataset_name and the img_path. Don't put a / between, because the img_path already contains that
        lung_images = [(default_datastore, f'processed_lungs/{dataset_name}{img_path}') for img_path in lung_images] # Make sure the paths are actual DataPaths
        
        random.seed(SEED) # Use the same random seed as I use and defined in the earlier cells
        random.shuffle(lung_images) # Shuffle the data so it's randomized
        
        ## Testing images
        amount_of_test_images = math.ceil(len(lung_images) * TRAIN_TEST_SPLIT_FACTOR) # Get a small percentage of testing images

        lung_test_images = lung_images[:amount_of_test_images]
        lung_training_images = lung_images[amount_of_test_images:]

        print(f'... we have {len(lung_test_images)} testing images and {len(lung_training_images)} training images.')
        
        # Add them all to the other ones
        testing_datapaths.extend(lung_test_images)
        training_datapaths.extend(lung_training_images)

        print(f'We already have {len(testing_datapaths)} testing images and {len(training_datapaths)} training images, on to process more lungs if necessary!')

    training_dataset = Dataset.File.from_files(path=training_datapaths)
    testing_dataset = Dataset.File.from_files(path=testing_datapaths)

    training_dataset = training_dataset.register(ws,
        name=os.environ.get('TRAIN_SET_NAME'), # Get from the environment
        description=f'The Lung Images to train, resized tot 128, 128',
        tags={'lungs': os.environ.get('LUNGS'), 'AI-Model': 'CNN', 'Split size': str(1 - TRAIN_TEST_SPLIT_FACTOR), 'type': 'training', 'GIT-SHA': os.environ.get('GIT_SHA')},
        create_new_version=True)

    print(f"Training dataset registered: {training_dataset.id} -- {training_dataset.version}")

    testing_dataset = testing_dataset.register(ws,
        name=os.environ.get('TEST_SET_NAME'), # Get from the environment
        description=f'The Lung Images to test, resized tot 128, 128',
        tags={'lungs': os.environ.get('LUNGS'), 'AI-Model': 'CNN', 'Split size': str(TRAIN_TEST_SPLIT_FACTOR), 'type': 'testing', 'GIT-SHA': os.environ.get('GIT_SHA')},
        create_new_version=True)

    print(f"Testing dataset registered: {testing_dataset.id} -- {testing_dataset.version}")

def main():
    ws = connectWithAzure()

    print('Processing the images')
    prepareDataset(ws)
    
    print('Splitting the images')
    trainTestSplitData(ws)

if __name__ == '__main__':
    main()