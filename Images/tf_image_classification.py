# Importing all the necessary librarires

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

from tensorflow.python.keras.engine.data_adapter import train_validation_split

# Download and cache dataset in ~/.keras/datasets

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# How many images are there?

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Let's extract the list of images of roses and open one

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0])).show()

# We can also extract a list of images of tulips

tulips = list(data_dir.glob('tulips/*'))


# Loading the images from directory into tf Dataset
# First define the batch_size and image size
batch_size = 32
img_height = 180
img_width = 180

# Make the dataset using image_dataset_from_directory
# We can use the same function for the training and validation datasets. 
# The seed needs to be the same and the subset argument is how we select which dataset we would like. 
# The image_size and batch_size arguments specify how the data is structured

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir
    , validation_split=0.2
    , subset="training"
    , seed=42
    , image_size=(img_height, img_width)
    , batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir
    , validation_split=0.2
    , subset="validation"
    , seed=42
    , image_size=(img_height, img_width)
    , batch_size=batch_size
)

# Retrieve the class names

class_names = train_ds.class_names
print(class_names)

# Let's plot some of the images to get an idea of what we are working with/validate that they look OK.

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# I'm not clear on the AUTOTUNE but maybe it dynamically changes the prefetch size based on performance/my system.

AUTOTUNE = tf.data.experimental.AUTOTUNE

# I am only 50% clear on what this is doing

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



num_classes=5

# Build our model sequentially:
#   Normalize Images with Rescaling layer
#   CNN with 3x3 kernel and 16 filters
#   Max Pooling with (2,2) pool size and (1,1) stride
#   CNN with 3x3 kernel and 32 filters
#   Max Pooling with (2,2) pool size and (1,1) stride
#   CNN with 3x3 kernel and 64 filters
#   Max Pooling with (2,2) pool size and (1,1) stride
#   Flatten
#   Dense with 128 output neurons
#   Dense with class logit outputs

model=Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)) # Exciting but expect it to change
    , layers.Conv2D(16, 3, padding='same', activation='relu')
    , layers.MaxPooling2D()
    , layers.Conv2D(32, 3, padding='same', activation='relu')
    , layers.MaxPooling2D()
    , layers.Conv2D(64, 3, padding='same', activation='relu')
    , layers.MaxPooling2D()
    , layers.Flatten()
    , layers.Dense(128, activation='relu')
    , layers.Dense(num_classes)
])

# Compile the model with ADAM optimizer and SparseCategoricalCrossentropy

model.compile(optimizer='adam'
                , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                , metrics=['accuracy'])

# Print model summary

model.summary()


# Train model over 5 epochs
epochs = 1

history = model.fit(
    train_ds
    , validation_data=val_ds
    , epochs=epochs
)

# Visualize the accuracy and loss over the course of training

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title("Training and Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc='upper right')
plt.title("Training and Validation Loss")
plt.show()

# Data augmentation to reduce overfitting
# Transform existing data into new but still representative data
# Randomly flip over the horizontal axis, rotate and zoom

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3))
        , layers.experimental.preprocessing.RandomRotation(0.1)
        , layers.experimental.preprocessing.RandomZoom(0.1)
    ]
)

# Display the effect of the data augmentation

plt.figure(figsize=(10,10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3,3,i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis('off')

# Next we are going to apply dropout to the network.
# During training we turn off certain neurons 

model=Sequential([
    data_augmentation
    , layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)) # Exciting but expect it to change
    , layers.Conv2D(16, 3, padding='same', activation='relu')
    , layers.MaxPooling2D()
    , layers.Conv2D(32, 3, padding='same', activation='relu')
    , layers.MaxPooling2D()
    , layers.Conv2D(64, 3, padding='same', activation='relu')
    , layers.MaxPooling2D()
    , layers.Dropout(0.2)
    , layers.Flatten()
    , layers.Dense(128, activation='relu')
    , layers.Dense(num_classes)
])

model.compile(optimizer='adam'
                , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                , metrics=['accuracy'])

model.summary()

epochs=15

history=model.fit(
    train_ds
    , validation_data=val_ds
    , epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()


sunflower_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg'
sunflower_path = tf.keras.utils.get_file("Red_sunflower", origin=sunflower_url)

img = keras.preprocessing.image.load_img(sunflower_path, target_size = (img_height, img_width))

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f'This image most likely belongs to {class_names[np.argmax(score)]} with a {100*np.max(score)} percent confidence')