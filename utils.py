import numpy as np
import tensorflow as tf
import os
import random
import cv2
from tensorflow.keras.applications.densenet import preprocess_input

def preprocess(x):
    # RGB to BGR
    img = tf.reverse(x, axis=[-1]) 
    img -= np.array([103.939, 116.779, 123.68])
    return img

def deprocess(x):
    # BGR to RGB
    img = x + np.array([103.939, 116.779, 123.68])
    img = tf.reverse(img, axis=[-1]) 
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def get_image(img_path, resize=True, shape=(256,256)):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(image, shape)
    image = image.astype(np.float32)
    return image

class ImageDataset:
    def __init__(self, dir_path, batch_size=8, file_type="jpg"):
        self.image_paths = self._find_images(dir_path, file_type)
        random.shuffle(self.image_paths)
        self.batch_size = batch_size
        self.ds_pointer = 0
        
    def _find_images(self, dir_path, file_type):
        image_paths = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                if f'.{file_type}' in file:
                    image_paths.append(os.path.join(root, file))
        
        return image_paths

    def get_batch(self):
        if self.ds_pointer + self.batch_size > len(self.image_paths):
            self.ds_pointer = 0
        
        images = []
        for i in range(self.batch_size):
            path = self.image_paths[self.ds_pointer + i]
            image = get_image(path)
            images.append(image)
        
        images = np.stack(images)
        images = preprocess(images)
        
        self.ds_pointer += self.batch_size
        return images


    def __len__(self):
        return len(self.image_paths)

