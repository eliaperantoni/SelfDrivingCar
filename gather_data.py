import cv2
import time
import os
import numpy as np
from utils.getkeys import key_check
from utils.grabscreen import grab_screen
from settings import getSet

sets = getSet()
file_name = sets.DEFAULT_TRAIN_FILE

WIDTH = sets.WIDTH
HEIGHT = sets.HEIGHT
CHANNELS = sets.CHANNELS


def keys_to_output(keys):
    #         A  W  D
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'W' in keys:
        output[1] = 1

    return output


def main(file_name):
    if True:  # Timeout
        for i in range(5)[::-1]:
            time.sleep(1)
            print(i + 1)

    last_time = time.time()

    if os.path.isfile(file_name):
        print("File exists")
        training_data = list(np.load(file_name))
    else:
        print("File does not exists")
        training_data = []

    while True:
        frame = grab_screen(region=(0, 40, 800, 640))
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        k = key_check()
        keys = keys_to_output(k)
        training_data.append([frame, keys])

        if len(training_data) % 500 == 0:
            print("Saving batch")
            np.save(file_name, training_data)

        # Stop recording when Z is pressed
        if 'Z' in k:
            print('Pause')
            np.save(file_name, training_data)
            while True:
                k = key_check()
                keys = keys_to_output(k)
                if 'X' in k:
                    print('Resume')
                    break
                elif 'Q' in k:
                    np.save(file_name, training_data)
                    return


if __name__ == "__main__":
    main(file_name)
# GTAV_HANDLE = cmdow /T
# Run cmdow GTAV_HANDLE /MOV -3 15
