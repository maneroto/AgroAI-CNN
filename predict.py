import os
import numpy as np
import pandas as pd
import keras.utils as image
from keras.models import load_model

from config.settings import MODEL_SAVE_PATH, MODEL_NAME, PREDICTION_MODEL, IMAGE_SIZE

def get_prediction_model():
  model_path = os.path.join(MODEL_SAVE_PATH, PREDICTION_MODEL)

  if not os.path.exists(model_path):
    print(f'The provided model does not exists, tried to load {model_path}')
    return
  
  model = load_model(model_path)
  return model

def get_image(image_path):
  img = image.load_img(image_path, target_size=IMAGE_SIZE)
  img = image.img_to_array(img)

  img = img / 255
  img = np.expand_dims(img, axis=0)

  return img

def get_classes():
  csv_save_name = f'{MODEL_NAME}-class_dict.csv'
  csv_path = os.path.join(MODEL_SAVE_PATH, csv_save_name)
  df = pd.read_csv(csv_path)
  return df

def get_predictions(image_path):
  model = get_prediction_model()
  img = get_image(image_path)

  predictions = model.predict(img) 
  top_index = np.argmax(predictions)

  return predictions, top_index

def main():
  image_path = './test_image.jpg'
  predictions, top_index = get_predictions(image_path)

  df = get_classes()
  class_name = df.loc[df['class_index'] == top_index, 'class'].values[0]

  print(predictions)
  print(f'The predicted class is: {class_name}')
  
if __name__ == '__main__':
  main()