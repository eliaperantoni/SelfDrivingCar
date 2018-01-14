import sys

sys.path.append('C:\\Users\\Elia\\PycharmProjects\\SelfDrivingGrandTheftAutoV')

import numpy as np
from v2.grabscreen import grab_screen
import cv2
import time
from directkeys import press_key, release_key, W, A, S, D
from keras.models import load_model
from v2.getkeys import key_check
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import random

WIDTH = 200
HEIGHT = 66
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'keras_models/v1.h5'

t_time = 0.09

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_optionsper_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


def straight():
    ##    if random.randrange(4) == 2:
    ##        ReleaseKey(W)
    ##    else:
    press_key(W)
    release_key(A)
    release_key(D)


def left():
    press_key(W)
    press_key(A)
    # ReleaseKey(W)
    release_key(D)
    # ReleaseKey(A)
    time.sleep(t_time)
    release_key(A)


def right():
    press_key(W)
    press_key(D)
    release_key(A)
    # ReleaseKey(W)
    # ReleaseKey(D)
    time.sleep(t_time)
    release_key(D)

for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)



model = load_model(MODEL_NAME)


def main():
    last_time = time.time()


    paused = False
    while (True):

        if not paused:
            # 800x600 windowed mode
            # screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0, 40, 800, 640))
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            screen = cv2.resize(screen, (200, 66))
            prediction = model.predict(screen.reshape(1, HEIGHT, WIDTH, 3))[0]
            print(prediction)

            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            else:
                straight()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                release_key(A)
                release_key(W)
                release_key(D)
                time.sleep(1)



main()
