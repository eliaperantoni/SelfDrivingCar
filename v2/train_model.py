import numpy as np
from alexnet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 1
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'. \
    format(LR, 'alexnet', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('training_data.npy')

train = train_data[:-100]
test = train_data[-100:]

train_x = np.array([i[0] for i in train]).reshape([-1, WIDTH, HEIGHT, 1]) # Prendo solo le immagini
train_y = np.array([i[1] for i in train]) # Prendo solo le label

test_x = np.array([i[0] for i in test]).reshape([-1, WIDTH, HEIGHT, 1]) # Prendo solo le immagini
test_y = np.array([i[1] for i in test]) # Prendo solo le label

model.fit({'input': train_x}, {'targets': train_y},
          n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, run_id=MODEL_NAME, show_metric=True)

model.save('models/model.tfl')

# tensorboard --logdir=C:/Users/Elia/PycharmProjects/SelfDrivingGrandTheftAutoV/v2/log