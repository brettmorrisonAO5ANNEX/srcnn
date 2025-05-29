import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers

def create_model(optimizer='adam', loss='mean_squared_error'):
    model = models.Sequential()
    # Input layer
    model.add(layers.Input(shape=(None, None, 3)))
    # Patch extraction and representation
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same'))
    # Non-linear mapping
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    # Reconstruction
    model.add(layers.Conv2D(3, (5, 5), activation='linear', padding='same'))

    # compile and return model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def test_model(img_path):
    return
    