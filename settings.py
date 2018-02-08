settings = {
    "WIDTH": 200,
    "HEIGHT": 66,
    "CHANNELS": 3,
    "BALANCE_MODE": 'RESAMPLE',
    "DEFAULT_TRAIN_FILE_DIRECTORY": 'train_data/env/raw/',
    "DEFAULT_TRAIN_FILE": 'train_data/env/raw/training_data_{}.npy',
    "DEFAULT_TRAIN_FILE_MERGED": 'train_data/env/training_data_merged.npy',
    "DEFAULT_TRAIN_FILE_BALANCED": 'train_data/env/training_data_balanced.npy',
    "LEARNING_RATE": 1e-3,
    "DROPOUT_RATE": 0.5,
    "EPOCHS": 8,
    "VALIDATION_SPLIT": 0.1,
    "BATCH_SIZE": 100,
    "DEFAULT_MODEL_FILE": 'saved_models/env/env_v2_non_balanced.h5'
}
