import numpy as np
from settings import settings
import glob


def merge(files):
    arr = [np.load(file) for file in files]
    print("Loaded all files, saving...")
    return np.vstack(arr)


def find_all(folder):
    return glob.glob(folder + "/*.npy")


if __name__ == '__main__':
    folder = settings["DEFAULT_TRAIN_FILE_DIRECTORY"]
    np.save(settings["DEFAULT_TRAIN_FILE_MERGED"], merge(find_all(folder)), allow_pickle=False)
