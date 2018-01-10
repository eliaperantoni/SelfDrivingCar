import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2 as cv

file_name = 'final_training_data.npy'


def display(train_data):
    for item in train_data:
        img = item[0]
        choice = item[1]
        # img = cv.resize(img, (160, 120))
        cv.imshow('window', img)
        print(choice)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


def balance(train_data, verbose=True):
    """
    :param train_data: Una lista del tipo [np.array(W*H), list(3,)]
    :param verbose: Boolean
    :return: Una lista come quella di partenza ma bilanciata tra W, A, D
    """
    if verbose: print('Before: ', Counter(pd.DataFrame(train_data)[1].apply(str)))
    left, right, forward = [], [], []
    shuffle(train_data)

    # Divido i vari esempi nelle varie direzioni
    for item in train_data:
        img = item[0]
        choice = item[1]

        if choice == [1, 0, 0]:
            left.append([img, choice])
        elif choice == [0, 0, 1]:
            right.append([img, choice])
        elif choice == [0, 1, 0]:
            forward.append([img, choice])

    min_length = min([len(left),
                      len(right),
                      len(forward)])

    left = left[:min_length]
    right = right[:min_length]
    forward = forward[:min_length]

    final_data = left + right + forward
    shuffle(final_data)
    if verbose: print('Before: ', Counter(pd.DataFrame(final_data)[1].apply(str)))
    return final_data


if __name__ == "__main__":
    train_data = np.load(file_name)
    #display(train_data)
    balanced_data = balance(train_data)
    np.save(file_name[:-4]+'_balanced.npy', balanced_data)
