import keras
import numpy as np
from models.nvidianet import nvidianet
import cv2 as cv
import sys, os
from settings import settings
import glob
import random


def generator(files):
    i = 0
    while True:
        if i == len(files):
            i = 0
            random.shuffle(files)
        sample = np.load(files[i])
        i += 1
        train_x = np.array([i[0] for i in sample]).reshape(-1, HEIGHT, WIDTH, CHANNELS)
        train_y = [float(i[1]["turn_rate"]) for i in sample]
        yield train_x, train_y


tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

WIDTH = settings["WIDTH"]
HEIGHT = settings["HEIGHT"]
LR = settings["LEARNING_RATE"]
EPOCHS = settings["EPOCHS"]
DROPOUT = settings["DROPOUT_RATE"]
CHANNELS = settings["CHANNELS"]
BATCH_SIZE = settings["BATCH_SIZE"]
VALIDATION_SPLIT = settings["VALIDATION_SPLIT"]

model = nvidianet(WIDTH, HEIGHT, CHANNELS, LR, DROPOUT)

files = glob.glob(settings["DEFAULT_TRAIN_FILE_PROCESSED_DIRECTORY"] + "*.npy")

model.fit_generator(generator(files), epochs=8, steps_per_epoch=len(files))

model.save(settings["DEFAULT_MODEL_FILE"])

# tensorboard --logdir=C:\Users\Elia\PycharmProjects\SelfDrivingGrandTheftAutoV\v2\Graph
