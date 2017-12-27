from PIL import ImageGrab
from directkeys import press_key, release_key, W, A, S, D
import numpy as np
import cv2 as cv
import time

vertices = np.array([[10, 500],
                     [10, 300],
                     [300, 200],
                     [500, 200],
                     [800, 300],
                     [800, 500]])


def roi(frame, vertices):
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, vertices, 255)
    masked = cv.bitwise_and(frame, mask)
    return masked


def process_frame(frame):
    processed_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    processed_frame = cv.Canny(processed_frame, threshold1=200, threshold2=300)
    return processed_frame


if __name__ == '__main__':
    while True:
        last_time = time.time()
        frame = ImageGrab.grab(bbox=(0, 40, 800, 640))
        frame = np.array(frame)
        frame = roi(frame, vertices)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.imshow('window', process_frame(frame))
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
