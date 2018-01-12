import keras
import numpy as np
from kerasnet import kerasnet

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 1

model = kerasnet(HEIGHT, WIDTH)

train_data = np.load('training_data/merged_training_data_balanced.npy')

train = train_data[:-100]
test = train_data[-100:]

train_x = np.array([i[0] for i in train]).reshape([-1, HEIGHT, WIDTH, 1]) # Prendo solo le immagini
train_y = np.array([i[1] for i in train]) # Prendo solo le label

test_x = np.array([i[0] for i in test]).reshape([-1, HEIGHT, WIDTH, 1]) # Prendo solo le immagini
test_y = np.array([i[1] for i in test]) # Prendo solo le label

model.fit(train_x, train_y, batch_size=100, epochs=EPOCHS, validation_split=0.1, callbacks=[tbCallBack])
model.save('keras_models/v2.h5')
print(model.evaluate(test_x, test_y))

# tensorboard --logdir=C:\Users\Elia\PycharmProjects\SelfDrivingGrandTheftAutoV\v2\v2_carsimenv\Graph