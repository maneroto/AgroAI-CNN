from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_ROOT = BASE_DIR / 'data'
TRAIN_URL = DATASET_ROOT / 'train'
TEST_URL = DATASET_ROOT / 'test'
VALIDATE_URL = DATASET_ROOT / 'validate'

DATASET_TMP_DIR_NAME = 'PlantVillage/color'
DATASET_TMP_DIR = BASE_DIR / DATASET_TMP_DIR_NAME

DATASET_RANDOM_SEED = 42

MODEL_NAME = 'agro_plant_disease'