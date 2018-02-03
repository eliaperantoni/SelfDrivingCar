# TODO Metti a posto questa schifezza

class SettingsSet:
    def __init__(self,
                 width,
                 height,
                 channels,
                 balance_mode,
                 default_train_file_directory,
                 default_train_file,
                 default_train_file_m,
                 default_train_file_b,
                 learning_rate,
                 dropout_rate,
                 epochs,
                 validation_split,
                 batch_size,
                 default_model_file):

        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = channels
        self.BALANCE_MODE = balance_mode
        self.DEFAULT_TRAIN_FILE_DIRECTORY = default_train_file_directory
        self.DEFAULT_TRAIN_FILE = default_train_file
        self.DEFAULT_TRAIN_FILE_M = default_train_file_m
        self.DEFAULT_TRAIN_FILE_B = default_train_file_b
        self.LR = learning_rate
        self.DROPOUT = dropout_rate
        self.EPOCHS = epochs
        self.VALIDATION_SPLIT = validation_split
        self.BATCH_SIZE = batch_size
        self.DEFAULT_MODEL_FILE = default_model_file


env = SettingsSet(width=200,
                  height=66,
                  channels=3,
                  balance_mode='RESAMPLE',
                  default_train_file_directory='train_data/env/raw/',
                  default_train_file='train_data/env/raw/training_data_{}.npy',
                  default_train_file_m='train_data/env/training_data_merged.npy',
                  default_train_file_b='train_data/env/training_data_balanced.npy',
                  learning_rate=1e-3,
                  dropout_rate=0.5,
                  epochs=8,
                  validation_split=0.1,
                  batch_size=100,
                  default_model_file='saved_models/env/env_v3.h5')

gta = SettingsSet(width=200,
                  height=66,
                  channels=3,
                  balance_mode='RESAMPLE',
                  default_train_file_directory='train_data/gta/raw/',
                  default_train_file='train_data/gta/training_data_{}.npy',
                  default_train_file_m='train_data/gta/training_data_merged.npy',
                  default_train_file_b='train_data/gta/training_data_balanced.npy',
                  learning_rate=1e-3,
                  dropout_rate=0.5,
                  epochs=8,
                  validation_split=0.1,
                  batch_size=100,
                  default_model_file='saved_models/gta/gta.h5')

ENVIRONMENTS = {'env': env, 'gta': gta}
ENVIRONMENT = 'env'


def getSet(environment=ENVIRONMENT):
    return ENVIRONMENTS[environment]
