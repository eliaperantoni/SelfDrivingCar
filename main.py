from PIL import ImageGrab
import numpy as np
import cv2 as cv
import time


def process_frame(frame):
    processed_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    processed_frame = cv.Canny(processed_frame, threshold1=200, threshold2=300)
    return processed_frame


if __name__ == '__main__':
    while True:
        last_time = time.time()
        frame = ImageGrab.grab(bbox=(0, 40, 800, 640))
        frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2RGB)
        cv.imshow('window', process_frame(frame))
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
