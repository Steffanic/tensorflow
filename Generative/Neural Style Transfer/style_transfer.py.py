# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


# %%
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


# %%
def tensor_to_image(tensor):
    tensor = tf.multiply(tensor, 255)
    print(f"{tensor.shape=}")
    tensor = np.array(tensor[0], dtype=np.uint8)
    if np.ndim(tensor)>3:
        print(f"{tensor.shape=}")
        assert tensor.shape[0] == 1
        tensor = tensor[0]
        print(tensor)
    return PIL.Image.fromarray(tensor)


# %%
content_path = './pat.jpg'
style_path = './el_huervo_style.jpg'


# %%
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# %%
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    
    plt.imshow(image)
    if title:
        plt.title(title)


# %%
content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')
print(content_image.shape)

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
print(style_image.shape)


# %%
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))
tensor_to_image(stylized_image)


# %%



