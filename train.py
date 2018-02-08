import keras
import numpy as np
from models.nvidianet import nvidianet
import cv2 as cv
import sys, os
from settings import settings

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)





WIDTH = settings["WIDTH"]
HEIGHT = settings["HEIGHT"]
LR = settings["LEARNING_RATE"]
EPOCHS = settings["EPOCHS"]
DROPOUT = settings["DROPOUT_RATE"]
CHANNELS = settings["CHANNELS"]
BACTH_SIZE = settings["BATCH_SIZE"]
VALIDATION_SPLIT = settings["VALIDATION_SPLIT"]

model = nvidianet(WIDTH, HEIGHT, CHANNELS, LR, DROPOUT)

train_data = np.load(settings["DEFAULT_TRAIN_FILE_BALANCED"])



train = train_data[:-100]
test = train_data[-100:]

train_x = np.array([i[0] for i in train]).reshape(-1, HEIGHT, WIDTH, CHANNELS)
train_y = [float(i[1]["turn_rate"]) for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, HEIGHT, WIDTH, CHANNELS)
test_y = [float(i[1]["turn_rate"]) for i in test]

model.fit(train_x, train_y, batch_size=BACTH_SIZE, epochs=EPOCHS,
          validation_split=VALIDATION_SPLIT, callbacks=[tbCallBack])

model.save(settings["DEFAULT_MODEL_FILE"])

# tensorboard --logdir=C:\Users\Elia\PycharmProjects\SelfDrivingGrandTheftAutoV\v2\Graph
