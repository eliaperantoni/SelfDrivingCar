import keras
import numpy as np
from kerasnet import kerasnet
import cv2 as cv

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 8

model = kerasnet(HEIGHT, WIDTH)

train_data = np.load('training_data_balanced.npy')

train = train_data[:-100]
test = train_data[-100:]

train_x = np.array([i[0] for i in train]).reshape(-1,HEIGHT,WIDTH,1)
train_y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,HEIGHT,WIDTH,1)
test_y = [i[1] for i in test]

model.fit(train_x, train_y, batch_size=100, epochs=8, validation_split=0.1, callbacks=[tbCallBack])

model.save('keras_models/v1.h5')

# tensorboard --logdir=C:\Users\Elia\PycharmProjects\SelfDrivingGrandTheftAutoV\v2\Graph