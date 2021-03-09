"""
The Fruits360 dataset is available at https://www.kaggle.com/moltean/fruits
"""

import numpy as np
import os
import PIL
import PIL.Image as Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

TRAINING_SET_PATH = "C:/Users/dxlat/Dataset/fruits-360/Training"
TEST_SET_PATH = "C:/Users/dxlat/Dataset/fruits-360/Test"
RANDOM_IMAGE_PATH = 'C:/Users/dxlat/Dataset/fruits-360/Test/Apple Braeburn/4_100.jpg'


class Fruits360Dataset:
    def __init__(self, config):
        self.config = config
        
        fruits_root = pathlib.Path(TRAINING_SET_PATH) #TODO Take path from config

        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))
        self.classes = [class_i for class_i in fruits_root.glob("*")]
        
        # Split training set into train n val
        list_ds = tf.data.Dataset.list_files(str(fruits_root/'*/*'))
        self.train_set = list_ds.map(self.process_path)

        # Import test set
        fruits_test_root = pathlib.Path(TEST_SET_PATH)
        self.test_set = tf.data.Dataset.list_files(str(fruits_test_root/'*/*')).map(self.process_path)

    def process_path(self, file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        return tf.io.read_file(file_path), label

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]

    def training_size(self):
        return len(self.train_set)

    def validation_size(self):
        return len(self.val_set)

    def test_size(self):
        return len(self.test_set)

    def validation_size(self):
        return len(self.val_set)

    def classes(self):
        return self.classes

    def show_random_image_from_test_set(self):
        Image.open().show(title='Pomme')


# TEST
# from utils.config import process_config
Fruits360Dataset(None).show_random_image_from_test_set(RANDOM_IMAGE_PATH)
# Fruits360Dataset(process_config('meta_fruits_train_config.json'))
