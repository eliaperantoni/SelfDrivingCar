import numpy as np
from settings import settings


file_name = settings["DEFAULT_TRAIN_FILE"]

WIDTH = settings["WIDTH"]
HEIGHT = settings["HEIGHT"]
CHANNELS = settings["WIDTH"]

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
