# Fruit Recognition using VGG16

A deep learning project that classifies 15 types of fruits from images using Transfer Learning with the VGG16 architecture. This project was built as a group assignment using Python, TensorFlow, and Keras.

---

## Team Members

| Member | Work Done |
|--------|-----------|
| Ayushman | Dataset collection and preparation, assisted in training and prediction, GitHub repository setup, deployment and README |
| Priyanshu | Model architecture design, VGG16 integration, training pipeline (train.py), train/validation split |
| Ankit | Prediction script (predict.py), model testing, output graphs analysis, documentation |

---

## About the Project

This project uses a pre-trained VGG16 model trained on ImageNet and fine-tunes it to recognize 15 different fruits. The top layers of VGG16 are replaced with custom dense layers and trained on a fruit dataset. Transfer learning allows the model to achieve good accuracy without training from scratch.

---

## Fruits It Can Recognize

| No. | Fruit | No. | Fruit |
|-----|-------|-----|-------|
| 1 | Apple Braeburn | 9 | Lychee |
| 2 | Avocado | 10 | Mango |
| 3 | Banana | 11 | Papaya |
| 4 | Cherry | 12 | Pineapple |
| 5 | Grape | 13 | Pomegranate |
| 6 | Guava | 14 | Strawberry |
| 7 | Kiwi | 15 | Watermelon |
| 8 | Lemon | | |

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- VGG16 (ImageNet weights)

---

## Project Structure

```
Fruit-Recognition-VGG16/
|
|-- dataset/
|   |-- Train/
|       |-- Apple Braeburn 1/
|       |-- Banana 1/
|       |-- ... (one folder per fruit)
|
|-- models/
|   |-- fruit_vgg16_model.h5
|
|-- outputs/
|   |-- accuracy.png
|   |-- loss.png
|
|-- train.py
|-- predict.py
|-- README.md
```

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Ayushman-codex/Fruit-Recognition-VGG16.git
cd Fruit-Recognition-VGG16
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy matplotlib
```

### 3. Add Dataset
Place your dataset inside the dataset/Train folder with one subfolder per fruit class.

### 4. Train the Model
```bash
python train.py
```

### 5. Predict a Fruit
Place your test image as test.jpg in the root folder and run:
```bash
python predict.py
```

Sample Output:
```
Predicted Fruit: Mango
Confidence: 97.84 %

Top 3 Predictions:
1. Mango - 97.84 %
2. Papaya - 1.52 %
3. Guava - 0.64 %
```

---

## How train.py Works

- Loads images from dataset/Train using ImageDataGenerator
- Applies rescaling (1/255) and splits data into 80% training and 20% validation
- Loads VGG16 pretrained model with ImageNet weights, top layers removed
- VGG16 layers are frozen so only custom layers are trained
- Adds Flatten, Dense(256, ReLU), Dropout(0.5), and final Softmax layer
- Trains for 5 epochs using Adam optimizer and categorical crossentropy loss
- Saves trained model to models/fruit_vgg16_model.h5
- Saves accuracy and loss graphs to outputs folder

## How predict.py Works

- Loads the saved model from models/fruit_vgg16_model.h5
- Reads test.jpg and resizes it to 224x224
- Converts image to array, expands dimensions, and normalizes pixel values
- Runs model.predict and finds the class with highest probability
- Prints predicted fruit name, confidence percentage, and top 3 predictions

---

## Model Architecture

```
VGG16 (pretrained, frozen)
        |
    Flatten
        |
  Dense (256, ReLU)
        |
  Dropout (0.5)
        |
  Dense (15, Softmax)
```

VGG16 base layers are kept frozen so that ImageNet features are preserved. A Dropout layer is added to reduce overfitting. The final Dense layer has 15 neurons for 15 fruit classes.

---

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Epochs | 5 |
| Batch Size | 32 |
| Input Image Size | 224 x 224 |
| Validation Split | 20% |

---

## Output Graphs and Result

After training, two graphs and one predicted result are saved in the outputs folder:

- accuracy.png - shows training and validation accuracy across epochs
- loss.png - shows training and validation loss across epochs
- screenshot.png - shows the predicted result of the image

---

## Note

Dataset is also included in this repository. Model file is uploaded using Git LFS. To download the model, make sure Git LFS is installed before cloning.

---

## Authors

- Ayushman - https://github.com/Ayushman-codex
- Priyanshu
- Ankit

## The model is trained on limited no. of dataset therefore it is not fully accurate with different types