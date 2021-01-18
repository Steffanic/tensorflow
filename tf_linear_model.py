import tensorflow as tf
import matplotlib.pyplot as plt

W_TRUE = 3.5
B_TRUE = 0.5

NUM_EXAMPLES = 1000

x = tf.random.normal(shape=[NUM_EXAMPLES])

noise = tf. random. normal(shape=[NUM_EXAMPLES])

y = x * W_TRUE + B_TRUE + noise

plt.scatter(x, y, c = 'y')
plt.show()



class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Initialize random guesses for weight and bias

        self.w = tf.Variable(tf.random.normal(shape=[1]))
        self.b = tf.Variable(tf.random.normal(shape=[1]))

    def __call__(self, x):
        return self.w * x + self.b


model = MyModel()

print(f"Variables: {model.variables}")

def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y-predicted_y))

plt.scatter(x, y, c='y')
plt.scatter(x, model(x), c = 'r')
plt.show()

print(f"Current loss: {loss(model(x), y).numpy()}")

def train(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))

    dw, db = t.gradient(current_loss, [model.w, model.b])

    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)

model = MyModel()

Ws, bs = [], []
epochs = range(20)

def training_loop(model, x, y):
    
    for epoch in epochs:
        train(model, x, y, learning_rate=0.2)

        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print(f'Epoch {epoch}: W = {Ws[-1]} b = {bs[-1]}, loss = {current_loss}')

print(f"Starting: W={model.w} b={model.b}, loss={loss(y, model(x))}")

training_loop(model, x, y)

plt.plot(epochs, Ws, "r", epochs, bs, "b")

plt.plot([W_TRUE] * len(epochs), "r--", [B_TRUE] * len(epochs), "b--")

plt.legend(["W", "b", "True W", "True b"])
plt.show()

plt.scatter(x, y, c='b')
plt.scatter(x, model(x), c='r')

plt.show()

print(f"Current loss: {loss(y, model(x)).numpy()}")