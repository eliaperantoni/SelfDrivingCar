import cv2
import time
import os
import numpy as np
from settings import getSet

sets = getSet()
file_name = sets.DEFAULT_TRAIN_FILE

WIDTH = sets.WIDTH
HEIGHT = sets.HEIGHT
CHANNELS = sets.CHANNELS

vars = {"iter": 0, "package": 0}
training_data = []

def save_data(frame, data):
    global training_data
    training_data.append([frame, data])
    vars["iter"] += 1
    if vars["iter"] >= 3000:
        np.save(file_name.format(vars["package"]), np.array(training_data))
        training_data = []
        print("Batch {} saved".format(vars["package"]))
        vars["package"] += 1
        vars["iter"] = 1
