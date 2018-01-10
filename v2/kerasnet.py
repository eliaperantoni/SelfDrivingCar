import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import BatchNormalization
from keras.layers import Dropout

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def kerasnet(width, height) -> keras.models.Sequential:
    model = Sequential()
    model.add(Conv2D(96, kernel_size=11, strides=4, activation='relu',
                     input_shape=[width, height, 1], padding='same'))
    model.add(MaxPooling2D(3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(384, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(384, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model