import keras
import numpy as np
from models.nvidianet import nvidianet
import cv2 as cv
import sys, os
from settings import getSet

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

sets = getSet()



WIDTH = sets.WIDTH
HEIGHT = sets.HEIGHT
LR = sets.LR
EPOCHS = sets.EPOCHS
DROPOUT = sets.DROPOUT
CHANNELS = sets.CHANNELS
BACTH_SIZE = sets.BATCH_SIZE
VALIDATION_SPLIT = sets.VALIDATION_SPLIT

model = nvidianet(WIDTH, HEIGHT, CHANNELS, LR, DROPOUT)

train_data = np.load(sets.DEFAULT_TRAIN_FILE_B)



train = train_data[:-100]
test = train_data[-100:]

train_x = np.array([i[0] for i in train]).reshape(-1, HEIGHT, WIDTH, CHANNELS)
train_y = [float(i[1]["turn_rate"]) for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, HEIGHT, WIDTH, CHANNELS)
test_y = [float(i[1]["turn_rate"]) for i in test]

model.fit(train_x, train_y, batch_size=BACTH_SIZE, epochs=EPOCHS,
          validation_split=VALIDATION_SPLIT, callbacks=[tbCallBack])

model.save(sets.DEFAULT_MODEL_FILE)

# tensorboard --logdir=C:\Users\Elia\PycharmProjects\SelfDrivingGrandTheftAutoV\v2\Graph
