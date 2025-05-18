import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
import tensorflow_datasets as tfds

def create_model():
    model = Sequential()
    model.add(Input(shape=(None, None, 3)))
    # Patch extraction and representation
    model.add(Conv2D(64, (9, 9), activation='relu', padding='same'))
    # Non-linear mapping
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    # Reconstruction
    model.add(Conv2D(3, (5, 5), activation='linear', padding='same'))

    # return model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def train_model(model, train_data, val_data, epochs=10):
    model.fit(train_data=train_data, val_data=val_data, epochs=epochs)

def test_model(img):
    return
    