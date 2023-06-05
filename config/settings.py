from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_ROOT = BASE_DIR / 'data'
TRAIN_URL = DATASET_ROOT / 'train'
TEST_URL = DATASET_ROOT / 'test'
VALIDATE_URL = DATASET_ROOT / 'validate'

DATASET_TMP_DIR_NAME = 'PlantVillage'
DATASET_TMP_DIR = BASE_DIR / DATASET_TMP_DIR_NAME / 'color'

DATASET_RANDOM_SEED = 42

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
IMAGE_COLOR_MODE = 'rgb'

MODEL_NAME = 'agro_plant_disease'
MODEL_TRAINING_OPTIONS = {
    'epochs': 2,
    'steps_per_epoch': 100,
    'validation_steps': 50,
    'data_batch_size': 40,
    'verbose': 1,
    'shuffle': False,
}
MODEL_SAVE_PATH = BASE_DIR / 'models'

PREDICTION_MODEL = 'agro_plant_disease-77.94.h5'