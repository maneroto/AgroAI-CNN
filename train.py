import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from config.settings import IMAGE_SIZE, IMAGE_CHANNELS, IMAGE_COLOR_MODE
from config.settings import TRAIN_URL, VALIDATE_URL, TEST_URL
from config.settings import MODEL_NAME, MODEL_TRAINING_OPTIONS, MODEL_SAVE_PATH

def get_scratch_model():
  print("Creating model...")
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(38, activation='softmax'))
  
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model

def get_stored_model():
  model_name = input(f'Give your model name (It must be stored in {MODEL_SAVE_PATH}): ')
  model_path = os.path.join(MODEL_SAVE_PATH, model_name)

  if not re.match(r".+\.h5$", model_name):
    print("The given file needs to be a .h5 extension")
    return
  
  if not os.path.exists(model_path):
    print("The path to your model does not exists")
    return
  
  else:
    print(f'Loading {model_name} file as a model...')
    model = load_model(model_path)
    return model


# Create and return the AI model
def get_model():
  answer = input("Do you want to train from scratch (s) or use the existing model (e)? ")
  if answer == "s":
    model = get_scratch_model()
  elif answer == "e":
    model = get_stored_model()
  else:
    print("Invalid input. Please enter s or e.")
    return
  
  print("Printing model summary...")
  model.summary()
  
  return model

# Create and return data generators
def get_data_generators():
  print("Loading data generators...")
  # Make data augmentation for the training dataset
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
  )
  
  # Normalize validation dataset
  validation_datagen = ImageDataGenerator(
    rescale=1./255
  )

  # Normalize test dataset
  test_datagen = ImageDataGenerator(
    rescale=1./255
  )
  
  # Create training images from corresponding folder
  train_generator = train_datagen.flow_from_directory(
    TRAIN_URL,
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    color_mode=IMAGE_COLOR_MODE,
    batch_size=MODEL_TRAINING_OPTIONS['data_batch_size'],
    class_mode='categorical'
  )
  
  # Create validation images from corresponding folder
  validation_generator = validation_datagen.flow_from_directory(
    VALIDATE_URL,
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    color_mode=IMAGE_COLOR_MODE,
    batch_size=MODEL_TRAINING_OPTIONS['data_batch_size'],
    class_mode='categorical'
  )

  # Create testing images from corresponding folder
  test_generator = test_datagen.flow_from_directory(
    TEST_URL,
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    color_mode=IMAGE_COLOR_MODE,
    batch_size=MODEL_TRAINING_OPTIONS['data_batch_size'],
    class_mode='categorical'
  )
  
  return train_generator, validation_generator, test_generator

# Train the model
def train_model(model, train_generator, validation_generator):
  print("Training the model...")
  return model.fit(
    train_generator,
    epochs=MODEL_TRAINING_OPTIONS['epochs'],
    steps_per_epoch=MODEL_TRAINING_OPTIONS['steps_per_epoch'],
    validation_data=validation_generator,
    validation_steps=MODEL_TRAINING_OPTIONS['validation_steps'],
    verbose=MODEL_TRAINING_OPTIONS['verbose'],
    shuffle=MODEL_TRAINING_OPTIONS['shuffle']
  )

def plot_training(hist):
  training_accuracy = hist.history['accuracy']
  training_loss = hist.history['loss']
  validation_accuracy = hist.history['val_accuracy']
  validation_loss = hist.history['val_loss']
  
  index_validation_loss = np.argmin(validation_loss)
  lowest_validation_loss = validation_loss[index_validation_loss]
  index_highest_accuracy = np.argmax(validation_accuracy)
  highest_accuracy = validation_accuracy[index_highest_accuracy]
  
  epochs = [i+1 for i in range(len(training_accuracy))]
  
  loss_label = f'best epoch= {str(index_validation_loss + 1)}'
  accuracy_label = f'best epoch= {str(index_highest_accuracy + 1)}'

  # Plot training history
  plt.figure(figsize= (20, 8))
  plt.style.use('fivethirtyeight')

  plt.subplot(1, 2, 1)
  plt.plot(epochs, training_loss, 'r', label= 'Training loss')
  plt.plot(epochs, validation_loss, 'g', label= 'Validation loss')
  plt.scatter(index_validation_loss + 1, lowest_validation_loss, s= 150, c= 'blue', label= loss_label)
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, training_accuracy, 'r', label= 'Training Accuracy')
  plt.plot(epochs, validation_accuracy, 'g', label= 'Validation Accuracy')
  plt.scatter(index_highest_accuracy + 1 , highest_accuracy, s= 150, c= 'blue', label= accuracy_label)
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.tight_layout
  plt.show()

def evaluate_model(model, train_generator, validation_generator, test_generator):
  print("Evaluating model...")
  evaluation_steps = test_generator.n // MODEL_TRAINING_OPTIONS['data_batch_size']

  print('Evaluating training data...')
  train_score = model.evaluate(train_generator, steps= evaluation_steps, verbose= 1)
  print('Evaluating validation data...')
  validation_score = model.evaluate(validation_generator, steps= evaluation_steps, verbose= 1)
  print('Evaluating testing data...')
  test_score = model.evaluate(test_generator, steps= evaluation_steps, verbose= 1)

  print("Train Loss: ", train_score[0])
  print("Train Accuracy: ", train_score[1])
  print('-' * 20)
  print("Validation Loss: ", validation_score[0])
  print("Validation Accuracy: ", validation_score[1])
  print('-' * 20)
  print("Test Loss: ", test_score[0])
  print("Test Accuracy: ", test_score[1])
  
  return test_score, validation_score, train_score

def save_model(model, class_dict, score):
  print('Saving model...')
  model_save_name = f'{MODEL_NAME}-{"%.2f" %round(score, 2)}.h5'
  model_path = os.path.join(MODEL_SAVE_PATH, model_save_name)
  model.save(model_path)

  print('Saving labels...')
  index_series = pd.Series(list(class_dict.values()), name= 'class_index')
  class_series = pd.Series(list(class_dict.keys()), name= 'class')
  class_df = pd.concat([index_series, class_series], axis= 1)
  csv_save_name = f'{MODEL_NAME}-class_dict.csv'
  csv_path = os.path.join(MODEL_SAVE_PATH, csv_save_name)
  class_df.to_csv(csv_path, index= False)

def main():
  print(f'Starting {MODEL_NAME} training execution...')
  start_time = time.time()

  model = get_model()
  
  train_generator, validation_generator, test_generator = get_data_generators()

  training_history = train_model(model, train_generator, validation_generator)
  plot_training(training_history)

  test_score, _, _ = evaluate_model(model, train_generator, validation_generator, test_generator)

  save_model(
    model=model,
    class_dict=train_generator.class_indices,
    score=test_score[1] * 100
  )
  
  end_time = time.time()
  total_time = end_time - start_time
  
  print(f"Process has finished in {total_time} seconds.")

if __name__ == "__main__":
  main()
