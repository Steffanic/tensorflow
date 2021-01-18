import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


(images, labels), _ = tf.keras.datasets.cifar10.load_data()

print(labels.shape)
labels = np.reshape(labels, (50000,))

label_trans = {0 : "airplane",1 : "automobile",2 : "bird",3 : "cat",4 : "deer",5 : "dog",6 : "frog",7 : "horse",8 : "ship",9 : "truck"}

images=images/255

print(np.min(images), np.max(images))
print(images[0])
plt.imshow(images[0])
plt.show()

print(images.shape)

training_iters = 10
learning_rate = 0.001
batch_size = 128

n_input = 32

n_classes = 10

x = tf.placeholder("float", [None, 28, 28, 3])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

init = tf.initializers.GlorotUniform()

weights = {
    'wc1': tf.Variable(init(shape=(3, 3, 3, 32)), name="W0"),
    'wc2': tf.Variable(init(shape=(3, 3, 32, 64)), name="W1"),
    'wc3': tf.Variable(init(shape=(3, 3, 64, 128)), name="W2"),
    'wd1': tf.Variable(init(shape=(4*4*128, 128)), name="W3"),
    'out': tf.Variable(init(shape=(128, n_classes)), name="W6"),
}

biases = {
    'bc1': tf.Variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.Variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.Variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.Variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.Variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}