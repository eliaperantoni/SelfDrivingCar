import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import press_key, release_key, W, A, S, D
from keras.models import load_model
from getkeys import key_check

import random

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'keras_models/v1.h5'

t_time = 0.09


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
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80, 60))
            prediction = model.predict(screen.reshape(1, 80, 60, 1))[0]
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
