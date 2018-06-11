import keras
from keras.models import Sequential
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dropout, Lambda, Dense, Flatten, Conv2D

def init():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

init()

def nvidianet(width, height, channels, lr=0.01, dropout=0.5) -> keras.models.Sequential:
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=[height, width, channels]))
    model.add(Conv2D(24, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr))

    return model