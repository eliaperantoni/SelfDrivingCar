class SettingsSet:
    def __init__(self, width, height, channels, balance_mode, default_train_file, default_train_file_b, learning_rate, dropout_rate,
                 epochs, validation_split, batch_size, default_model_file):
        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = channels
        self.BALANCE_MODE = balance_mode
        self.DEFAULT_TRAIN_FILE = default_train_file
        self.DEFAULT_TRAIN_FILE_B = default_train_file_b
        self.LR = learning_rate
        self.DROPOUT = dropout_rate
        self.EPOCHS = epochs
        self.VALIDATION_SPLIT = validation_split
        self.BATCH_SIZE = batch_size
        self.DEFAULT_MODEL_file = default_model_file


env = SettingsSet(200, 66, 3, 'RESAMPLE', 'train_data/env/training_data.npy', 'train_data/env/training_data_balanced.npy',
                  1e-3, 0.5, 8, 0.1, 100, 'saved_models/env/env_v3.h5')
gta = SettingsSet(200, 66, 3, 'RESAMPLE', 'train_data/gta/training_data.npy', 'train_data/gta/training_data_balanced.npy',
                  1e-3, 0.5, 8, 0.1, 100, 'saved_models/gta/gta.h5')

ENVIRONMENTS = {'env': env, 'gta': gta}
ENVIRONMENT = 'env'


def getSet(environment=ENVIRONMENT):
    return ENVIRONMENTS[environment]
