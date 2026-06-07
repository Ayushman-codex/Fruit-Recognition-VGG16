import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("models/fruit_vgg16_model.h5")

img_path = "test.jpg"

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

class_names = [
    "Apple Braeburn 1",
    "Avocado 1",
    "Banana 1",
    "Cherry 1",
    "Grape 1",
    "Guava 1",
    "Kiwi 1",
    "Lemon 1",
    "Lychee 1",
    "Mango 1",
    "Papaya 1",
    "Pineapple 1",
    "Pomegranate 1",
    "Strawberry 1",
    "Watermelon 1"
]

print("Predicted Fruit:", class_names[np.argmax(prediction)])