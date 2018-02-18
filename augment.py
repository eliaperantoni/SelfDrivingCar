from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib.pyplot as plt
import numpy as np
from settings import settings
import glob, random

seq = iaa.Sequential([
    iaa.Dropout(0.0, 0.2),
    iaa.Crop(px=(0, 20)),
    iaa.Pad(px=(0, 20)),
])

if __name__ == "__main__":
    files = glob.glob(settings["DEFAULT_TRAIN_FILE_DIRECTORY"] + "*.npy")
    img = np.load(files[random.randint(0, len(files))])[random.randint(0, 1000)][0]
    img = seq.augment_image(img)
    plt.imshow(img)
    plt.show()