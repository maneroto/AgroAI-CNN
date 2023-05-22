import numpy as np
from keras.models import load_model
import keras.utils as image

from config.settings import MODEL_NAME

model = load_model(f'{MODEL_NAME}.h5')

img = image.load_img('test_image.jpg', target_size=(224, 224))
img = image.img_to_array(img)

img = img / 255
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
index = np.argmax(pred)

print(f'The predicted class is: {index}')