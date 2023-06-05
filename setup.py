import os
import time
import kaggle
import shutil
import dotenv
from sklearn.model_selection import train_test_split

from config.settings import *

def authenticate():
  try:
    kaggle.api.authenticate()
  except Exception as _:
    dotenv.load_dotenv()
    kaggle.api.authenticate_from_env()

def create_dataset_folders():
  print('Setting dataset folders...')
  if not os.path.exists(DATASET_ROOT):
    os.mkdir(DATASET_ROOT)
  if not os.path.exists(TRAIN_URL):
    os.mkdir(TRAIN_URL)
  if not os.path.exists(TEST_URL):
    os.mkdir(TEST_URL)
  if not os.path.exists(VALIDATE_URL):
    os.mkdir(VALIDATE_URL)

def create_dataset_folder_info(dataset, folder_dir, folder_name):
  print(f'Creating {folder_name} dataset...')
  for file in dataset:
    class_folder = os.path.split(file)[0]
    os.makedirs(os.path.join(folder_dir, class_folder), exist_ok=True)
    shutil.copy(os.path.join(DATASET_TMP_DIR, file), os.path.join(folder_dir, file))

def load_image_dataset():
  files = []

  if not os.path.exists(DATASET_TMP_DIR):
    print('Downloading dataset files...')

    kaggle_folder_name = './plantvillage dataset'
    kaggle.api.dataset_download_files('abdallahalidev/plantvillage-dataset', path='.', unzip=True)

    shutil.rmtree(os.path.join(kaggle_folder_name, 'segmented'))
    shutil.rmtree(os.path.join(kaggle_folder_name, 'grayscale'))

    os.rename(kaggle_folder_name, DATASET_TMP_DIR_NAME)

  for folder in os.listdir(DATASET_TMP_DIR):
    for file in os.listdir(os.path.join(DATASET_TMP_DIR, folder)):
      files.append(os.path.join(folder, file))

  train_files, test_files = train_test_split(files, test_size=0.2, shuffle=True, random_state=DATASET_RANDOM_SEED)
  validate_files, test_files = train_test_split(test_files, test_size=0.5, shuffle=True, random_state=DATASET_RANDOM_SEED)

  create_dataset_folders()

  create_dataset_folder_info(train_files, TRAIN_URL, "training")
  create_dataset_folder_info(test_files, TEST_URL, "testing")
  create_dataset_folder_info(validate_files, VALIDATE_URL, "validation")

def main():
  print("Starting files setup...")
  start_time = time.time()

  authenticate()
  load_image_dataset()

  end_time = time.time()
  total_time = end_time - start_time

  print(f"Process has finished in {total_time} seconds.")

if __name__ == "__main__":
  main()