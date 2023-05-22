from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from config.settings import TRAIN_URL, VALIDATE_URL, MODEL_NAME

# Define CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(38, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Make data augmentation for the training dataset
train_datagen =  ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Normalize validation dataset
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

# Create training images from corresponding folder
train_generator = train_datagen.flow_from_directory(
    TRAIN_URL,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Create validation images from corresponding folder
validation_generator = validation_datagen.flow_from_directory(
    VALIDATE_URL,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

model.save(f'{MODEL_NAME}.h5')