from settings import settings
from keras.models import load_model
import tensorflow as tf, numpy as np, glob
from keras.backend.tensorflow_backend import set_session
import random

WIDTH = settings["WIDTH"]
HEIGHT = settings["HEIGHT"]
CHANNELS = settings["CHANNELS"]
MODEL_NAME = settings["DEFAULT_MODEL_FILE"]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model = load_model(MODEL_NAME)


def main():
    files = glob.glob(settings["DEFAULT_TRAIN_FILE_PROCESSED_DIRECTORY"] + "*.npy")
    v = np.load(random.choice(files))
    test_x = np.array([i[0] for i in v]).reshape(-1, HEIGHT, WIDTH, CHANNELS)
    test_y = [float(i[1]["turn_rate"]) for i in v]
    print(model.evaluate(test_x, test_y))

main()
