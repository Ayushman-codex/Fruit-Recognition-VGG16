
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt
 
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
 
train_data = train_datagen.flow_from_directory(
    "dataset/Train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
 
val_data = train_datagen.flow_from_directory(
    "dataset/Train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
 
print("Classes Found:")
print(train_data.class_indices)
 
print("Number of Classes:", train_data.num_classes)
 
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
 
base_model.trainable = False
 
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])
 
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
 
model.summary()
 
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)
 
model.save("models/fruit_vgg16_model.h5")
 
print("Model Saved Successfully!")
 
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/accuracy.png')
plt.show()
 
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/loss.png')
plt.show()
