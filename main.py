from PIL import ImageGrab
import numpy as np
import cv2 as cv
import time

if __name__ == '__main__':
    while True:
        last_time = time.time()
        frame = ImageGrab.grab(bbox=(0, 40, 800, 640))
        frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2RGB)
        cv.imshow('window', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
        print('Loop took ', time.time() - last_time)
