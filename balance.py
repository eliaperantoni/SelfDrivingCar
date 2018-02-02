import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2 as cv
from sklearn.utils import resample
from sklearn.utils import resample
from settings import getSet

sets = getSet()

file_name = sets.DEFAULT_TRAIN_FILE_M

BALANCE_MODE = sets.BALANCE_MODE


def display(train_data):
    for item in train_data:
        img = item[0]
        choice = item[1]
        # img = cv.resize(img, (160, 120))
        cv.imshow('window', img)
        print(choice)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


def balance(train_data, verbose=True):
    pass # TODO Implementa la funzione di bilanciamento


if __name__ == "__main__":
    disp = input('Display? [y/N]\n')
    train_data = np.load(file_name)
    if disp == 'y':
        display(train_data)
    else:
        balanced_data = balance(train_data)
        np.save(file_name[:-4] + '_balanced.npy', balanced_data)
