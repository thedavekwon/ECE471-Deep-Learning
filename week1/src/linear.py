#!/bin/python3.6

import numpy as np
import tensorflow as tf

from tqdm import trange

NUM_FEATURES = 4
NUM_SAMP = 50
BATCH_SIZE = 32
NUM_BATCHES = 300
LEARNING_RATE = 0.1


class Data(object):
    def __init__(self, num_features=NUM_FEATURES, num_samp=NUM_SAMP):
        """
        Draw random weights and bias. Project vectors in R^NUM_FEATURES
        onto R with said weights and bias.
        """
        num_samp = NUM_SAMP
        sigma = 0.1
        np.random.seed(31415)

        # We're going to learn these paramters
        self.w = np.random.randint(low=0, high=5, size=(num_features, 1))
        self.b = 2

        self.index = np.arange(num_samp)
        self.x = np.random.uniform(size=(num_samp, num_features))
        self.y = self.x @ self.w + self.b + sigma * np.random.normal()

    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()


class Model(tf.Module):
    def __init__(self, num_features=NUM_FEATURES):
        """
        A plain linear regression model with a bias term
        """
        self.w = tf.Variable(tf.random.normal(shape=[num_features, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))

    def __call__(self, x):
        return tf.squeeze(x @ self.w + self.b)


if __name__ == "__main__":
    data = Data()
    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch()
            y_hat = model(x)
            loss = tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    w = np.squeeze(data.w)
    w_hat = np.squeeze(model.w.numpy())

    # print out true values versus estimates
    print("w, w_hat")
    for a, b in zip(w, w_hat):
        print(f"{a}, {b:0.2f}")
