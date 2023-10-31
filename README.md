# Plant Disease Classification Project Documentation

## Introduction

This document guides you through the process of running a plant disease classification model using TensorFlow and Keras. The model is built upon the ResNet50 architecture, a powerful convolutional neural network. Due to various constraints, I am unable to upload the trained model file in HDF5 (.h5) earlier trained format. However, I have refactored the code to save the model in TensorFlow's native format. Below, you will find a step-by-step guide on how to run the code and train the model yourself. 

## Prerequisites

Before you begin, ensure you have the following installed:
- Python (version 3.6 or newer)
- TensorFlow (version 2.0 or newer)
- Keras
- NumPy

You can install these packages using pip:

```bash
pip install tensorflow keras numpy
```

## Step 1: Download the Dataset

Download the PlantVillage dataset from the following link: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

Once downloaded, extract the dataset to a directory on your machine.

## Step 2: Set Up Your Python Environment

Create a new Python script or Jupyter notebook in your preferred IDE or text editor.

## Step 3: Import Libraries

In your script, start by importing the required libraries:

```python
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
```

## Step 4: Build the Model

Next, define and compile your model:

```python
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(38, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

## Step 5: Prepare the Data

Set up your data generators:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'path_to_your_dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

Ensure that you replace `'path_to_your_dataset'` with the path to your extracted PlantVillage dataset.

## Step 6: Train the Model

Now, you are ready to train your model:

```python
model.fit(train_generator, epochs=100, steps_per_epoch=len(train_generator))
```

## Step 7: Save the Model

After training, save your model in TensorFlow format:

```python
model.save("path_to_save_directory", save_format="tf")
```

Replace `"path_to_save_directory"` with the directory where you want to save your trained model.

## Step 8: Contact for Trained Model

If you require the trained model in HDF5 (.h5) format, feel free to contact me, and I will be happy to share it with you.

---

Follow these steps to train your own plant disease classification model using TensorFlow and Keras. If you encounter any issues or have questions, do not hesitate to reach out.