
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
 
model = tf.keras.models.load_model("models/fruit_vgg16_model.h5")
 
class_names = [
    "Apple Braeburn",
    "Avocado",
    "Banana",
    "Cherry",
    "Grape",
    "Guava",
    "Kiwi",
    "Lemon",
    "Lychee",
    "Mango",
    "Papaya",
    "Pineapple",
    "Pomegranate",
    "Strawberry",
    "Watermelon"
]
 
img_path = "test.jpg"
 
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0
 
prediction = model.predict(img_array)
 
predicted_index = np.argmax(prediction)
predicted_fruit = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100
 
print("Predicted Fruit:", predicted_fruit)
print("Confidence:", round(confidence, 2), "%")
 
print("\nTop 3 Predictions:")
top3 = np.argsort(prediction[0])[::-1][:3]
for i, idx in enumerate(top3):
    print(str(i + 1) + ".", class_names[idx], "-", round(prediction[0][idx] * 100, 2), "%")
