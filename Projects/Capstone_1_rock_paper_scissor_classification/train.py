import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import time
import random

# Config
input_size = 150
dataset_path = 'rock_paper_scissors_dataset'
dataset_train_path = os.path.join(dataset_path, 'train')
classes = ['rock', 'paper', 'scissors']
no_display_images = 5

# Image generators
train_val_gen = ImageDataGenerator(
                rescale=1./255,
                #preprocessing_function=preprocess_input,
                rotation_range=10,
                width_shift_range=10,
                height_shift_range=10,
                shear_range=10,
                zoom_range=0.2,
                horizontal_flip=False,
                vertical_flip=False,
                validation_split=0.2)

train_ds = train_val_gen.flow_from_directory('./rock_paper_scissors_dataset/train', 
                                         target_size=(input_size,input_size),
                                         batch_size=32,
                                         class_mode='categorical',
                                         color_mode='grayscale',
                                         #color_mode='rgb',
                                        subset='training'
                                        )

#val_gen = ImageDataGenerator(rescale=1./255)

val_ds = train_val_gen.flow_from_directory('./rock_paper_scissors_dataset/train', 
                                         target_size=(input_size,input_size),
                                         batch_size=1,
                                         shuffle=False,
                                         class_mode='categorical',
                                         color_mode='grayscale',
                                         #color_mode='rgb'
                                        subset='validation'
                                        )

#test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(rescale=1./255)
test_ds = test_gen.flow_from_directory('./rock_paper_scissors_dataset/test', 
                                         target_size=(input_size,input_size),
                                         shuffle=False,
                                         batch_size=1,
                                         class_mode='categorical',
                                         color_mode='grayscale'
                                         #  color_mode='rgb'
                                           )    

print(train_ds.class_indices)
X, y = next(train_ds)
X.shape
#preprocess_input(X).shape

# Create model function
def make_model(base_model_type = 'custom', 
               input_size = 150, 
               learning_rate = 1e-3, 
               num_classes = 2, 
               size_dense_1 = 256,
               size_dense_2 = 64, 
               drop_rate=0.2):
    
    inputs = keras.Input(shape=(input_size,input_size,1))

    if base_model_type == 'xception':
        base_model = Xception(weights="imagenet",
                         include_top=False,
                         input_shape=(input_size,input_size,3))
        base_model.trainable = False
        inputs = keras.Input(shape=(input_size,input_size,3))
        base = base_model(inputs)
        vectors = keras.layers.GlobalAveragePooling2D()(base)
        inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
        drop = keras.layers.Dropout(drop_rate)(inner)
        outputs = keras.layers.Dense(num_classes)(drop)
        if num_classes <= 2:
            loss = keras.losses.BinaryCrossentropy(from_logits=True)
        elif num_classes > 2:
            loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    elif base_model_type == 'custom':
        conv2d_1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu')(inputs)
        maxpool2d_1 = keras.layers.MaxPooling2D(
            pool_size=(2, 2))(conv2d_1)
        dropout_1 = keras.layers.Dropout(drop_rate)(maxpool2d_1)
        conv2d_2 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu')(dropout_1)
        maxpool2d_2 = keras.layers.MaxPooling2D(
            pool_size=(2, 2))(conv2d_2)
        dropout_2 = keras.layers.Dropout(drop_rate)(maxpool2d_2)
        conv2d_3 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu')(dropout_2)
        maxpool2d_3 = keras.layers.MaxPooling2D(
            pool_size=(2, 2))(conv2d_3)
        dropout_3 = keras.layers.Dropout(drop_rate)(maxpool2d_3)
        flatten_1 = tf.keras.layers.Flatten()(dropout_3)
        dense_1 = tf.keras.layers.Dense(size_dense_1, activation='relu', kernel_regularizer='l2')(flatten_1)
        dense_2 = tf.keras.layers.Dense(size_dense_2, activation='relu', kernel_regularizer='l2')(dense_1)

        #outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense_1)
        outputs = tf.keras.layers.Dense(num_classes, kernel_regularizer='l2', activation='softmax')(dense_2)
        #loss = keras.losses.BinaryCrossentropy()
        loss = keras.losses.CategoricalCrossentropy()
    else:
        print("Model not implemented.")
        return None
    
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    #optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.8)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model

# Select model from experiments data
input_size = 150
size_dense_1 = 128
size_dense_2 = 32
drop_rate = 0.2
learning_rate = 0.001
version = 23

# Create model
model = make_model(base_model_type='custom', 
                   num_classes=3, 
                   input_size=input_size,
                  size_dense_1 = size_dense_1,
                  size_dense_2 = size_dense_2,
                  drop_rate = drop_rate,
                  learning_rate = learning_rate)
print(model.summary())

# Define callbacks
checkpoint_string = f"rock_paper_scissors_v{version}_"+"{epoch:02d}_{val_accuracy:.3f}.h5"
checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_string, 
               save_best_only=True,
              monitor='val_accuracy',
              mode='max')

# Train model
start_training = time.time()
history = model.fit(train_ds, 
    epochs=10, 
    validation_data=val_ds,
   callbacks=[checkpoint_cb])
training_duration_min = (time.time() - start_training) / 60
print(f"Training took {training_duration_min} seconds.")

# Plot training/validation accuracy
plt.plot(history.history['accuracy'], label="training acc")
plt.plot(history.history['val_accuracy'], label="validation acc")
plt.title(f"V{version}: Training/validation accuracy")
plt.legend()
plt.show()

# Plot training/validation loss
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.title(f"V{version}: Training/validation loss")
plt.legend()
plt.show()

# Evaluate model
evaluation = model.evaluate(test_ds)
test_acc = evaluation[1]
print(test_acc)

# Save model as SavedModel
model.save(f'rock_paper_scissors_model/{version}')
# Save model as h5
model.save(f'rock_paper_scissors_model/rock_paper_scissors_model_{version}.h5')