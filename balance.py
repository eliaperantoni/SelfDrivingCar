import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2 as cv
from sklearn.utils import resample
from sklearn.utils import resample
from settings import settings
import glob, random

TRESHOLD = 0.075

def balance(train_data, verbose=True):
    shuffle(train_data)
    # TODO Bilancia
    return train_data


if __name__ == "__main__":
    x = 0
    files = glob.glob(settings["DEFAULT_TRAIN_FILE_DIRECTORY"] + "*.npy")
    for completition, file in enumerate(files):
        v = np.load(file)
        # left, forward, right = [], [], []
        # for item in v:
        #     if -TRESHOLD < item[1]["turn_rate"] < TRESHOLD:
        #         forward.append(item)
        #     elif item[1]["turn_rate"] < -TRESHOLD:
        #         left.append(item)
        #     else:
        #         right.append(item)
        # print(len(left), len(forward), len(right))
        # n = max((len(left), len(right)))
        # while len(left) < n:
        #     if(len(left)>0):
        #         left.append(random.choice(left))
        #     else:
        #         break
        # while len(right) < n:
        #     if (len(right) > 0):
        #         right.append(random.choice(right))
        #     else:
        #         break
        # print(len(left), len(forward), len(right))
        # v = np.array(left+forward+right)
        mirror = np.array([(cv.flip(elem[0], 1), {"turn_rate": -elem[1]["turn_rate"]}) for elem in v])
        np.random.shuffle(v)
        np.random.shuffle(mirror)
        merged = np.vstack((v, mirror))
        merged_split = np.array_split(merged, len(merged) // settings["BATCH_SIZE"])
        y = 0
        for batch in merged_split:
            for elem in batch:
                if elem[0].shape != (settings["HEIGHT"], settings["WIDTH"], settings["CHANNELS"]):
                    raise Exception("Dimensione non valida", elem[0].shape)
            np.save(settings["DEFAULT_TRAIN_FILE_PROCESSED"].format(x, y), batch)
            y += 1
        #np.save(settings["DEFAULT_TRAIN_FILE_PROCESSED_NORM"].format(i), v)
        #np.save(settings["DEFAULT_TRAIN_FILE_PROCESSED_MIRR"].format(i), mirror)
        print(round(completition/len(files)*100, 1))
        x += 1

