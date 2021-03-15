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
        
        fruits_root = pathlib.Path(self.config.training_set_path)

        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))
        self.classes = [class_i for class_i in fruits_root.glob("*")]
        
        # Split training set into train n val
        list_ds = tf.data.Dataset.list_files(str(fruits_root/'*/*'))
        self.train_set = list_ds.map(self.process_path)
        self.train_set.shuffle(buffer_size=100)
        val_set_size = int(0.85 * len(self.train_set))
        self.val_set = self.train_set.take(val_set_size)
        self.train_set = self.train_set.skip(val_set_size).take(len(self.train_set) - val_set_size)

        self.train_set = self.train_set.batch(self.config.batch_size)
        # print(f"Train shape: {self.train_set.batch(4).shape}")

        # Import test set
        fruits_test_root = pathlib.Path(self.config.test_set_path)
        self.test_set = tf.data.Dataset.list_files(str(fruits_test_root/'*/*')).map(self.process_path)

    def process_path(self, file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        return tf.io.read_file(file_path), label

    def next_batch(self, batch_size):
        idx = np.random.choice(self.training_size(), batch_size)
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
        Image.open(self.config.random_image_path).show(title='Pomme')

