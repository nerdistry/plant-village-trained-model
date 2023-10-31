import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

print(tf.__version__)

base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding a dropout layer
x = Dense(512, activation='relu')(x)  # Additional 512-unit Dense layer
predictions = Dense(38, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'plantvillage dataset/color',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.fit(train_generator, epochs=100, steps_per_epoch=len(train_generator))

model.save("path_to_save_directory", save_format="tf")
