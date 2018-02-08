import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2 as cv
from sklearn.utils import resample
from sklearn.utils import resample
from settings import settings
import glob


def balance(train_data, verbose=True):
    shuffle(train_data)
    # TODO Bilancia
    return train_data


if __name__ == "__main__":
    i = 0
    files = glob.glob(settings["DEFAULT_TRAIN_FILE_DIRECTORY"] + "*.npy")
    for completition, file in enumerate(files):
        v = np.load(file)
        mirror = np.array([(cv.flip(elem[0], 1), {"turn_rate": -elem[1]["turn_rate"]}) for elem in v])
        np.random.shuffle(v)
        np.random.shuffle(mirror)
        np.save(settings["DEFAULT_TRAIN_FILE_PROCESSED_NORM"].format(i), v)
        np.save(settings["DEFAULT_TRAIN_FILE_PROCESSED_MIRR"].format(i), mirror)
        print(round(completition/len(files)*100, 1))
        i += 1

