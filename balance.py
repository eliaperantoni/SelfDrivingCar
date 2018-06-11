import numpy as np
from collections import Counter
from random import shuffle
import cv2 as cv
from sklearn.utils import resample
from sklearn.utils import resample
from settings import settings
import glob, random
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

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
        for item in v:
            if item[1]["camera"] == "left":
                item[1]["turn_rate"] += settings["CORRECTION"]
            elif item[1]["camera"] == "right":
                item[1]["turn_rate"] -= settings["CORRECTION"]
            if item[1]["turn_rate"] > 1.0:
                item[1]["turn_rate"] = 1.0
            elif item[1]["turn_rate"] < -1.0:
                item[1]["turn_rate"] = -1.0
        mirror = np.array([(cv.flip(elem[0], 1), {"turn_rate": -elem[1]["turn_rate"]}) for elem in v])
        merged = np.vstack((v, mirror))
        images = np.array([item[0] for item in merged], dtype="uint8")
        # for i, image in enumerate(images):
        #     modified = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        #     images[i] = modified
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 8)),
            iaa.Add(value=(0, 45)),
        ])
        images = seq.augment_images(images)
        assert len(images) == len(merged)
        merged = np.array([(images[i], merged[i][1]) for i in range(len(merged))])
        merged_split = np.array_split(merged, len(merged) // settings["BATCH_SIZE"])
        y = 0
        for batch in merged_split:
            for elem in batch:
                if elem[0].shape != (settings["HEIGHT"], settings["WIDTH"], settings["CHANNELS"]):
                    raise Exception("Dimensione non valida", elem[0].shape)
            np.save(settings["DEFAULT_TRAIN_FILE_PROCESSED"].format(x, y), batch)
            y += 1
        print(round(completition/len(files)*100, 1))
        x += 1

