#! /usr/bin/env python3

from keras.datasets import mnist
import numpy as np
import random
from PIL import Image, ImageOps


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def convert_mnist_data(Mnist_X, Mnist_Y):
    """
    MNIST data from keras comes in the following format:
        X: shape is num_imagesx28x28. Image pixels are 0-255.
        y: shape is num_images. Value of each is the actual digit label (e.g. '5').
    We would like to use the data like this:
        training_data: List of num_images tuples with elements:
          x: flattened 784x1 input image, pixel values normalized to 0.0-1.0
          y: 10x1 label vector (one-hot encoded with the correct label)
    """
    # flatten the images into 784x1 arrays
    N = Mnist_X.shape[0]
    X = Mnist_X.reshape(N, 784, 1)

    # Convert from 0-255 to 0.0-1.0
    X = X / 255.0

    data = []
    for i in range(Mnist_X.shape[0]):
        x = X[i, ...]
        y = np.zeros((10, 1))
        y[Mnist_Y[i]] = 1.0
        data.append((x, y))
    return data


class FfNet:

    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(
            layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, train_data, num_epochs, mini_batch_size, eta, test_data=None):
        N = len(train_data)  # size of training data
        if test_data:
            N_test = len(test_data)
        assert N % mini_batch_size == 0, "Size of test data must be a multiple of minibatch size"
        for j in range(num_epochs):
            # shuffle the training data
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size]
                            for k in range(0, N, mini_batch_size)]
            for mini_batch in mini_batches:
                self.run_mini_batch(mini_batch, eta)
            if N_test:
                print("Epoch {}: {} / {}".format(j,
                                                 self.evaluate(test_data), N_test))
            else:
                print("Epoch {} complete".format(j))
        return 0

    def run_mini_batch(self, mini_batch, eta):
        N = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/N) * nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/N) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward pass
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(
            activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        # return np.count_nonzero(x == y)
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_image(self, x):
        return self.feedforward(x)


(train_X, train_y), (test_X, test_y) = mnist.load_data()

# # Convert the training data into a list of tuples of (X, Y)
train_data = convert_mnist_data(train_X, train_y)
test_data = convert_mnist_data(test_X, test_y)

# net = FfNet([784, 32, 10])
# net.SGD(train_data, 30, 10, 3.0, test_data=test_data)


weights = np.load('weights.npy', allow_pickle=True)
biases = np.load('biases.npy', allow_pickle=True)


with Image.open("eight.jpg") as im:
    im = ImageOps.grayscale(im)
    im = ImageOps.invert(im)
    im.show()
    im_np = np.array(im)
    im_np = im_np / 255.0

a = im_np.reshape(784, 1)
for b, w in zip(biases, weights):
    a = sigmoid(w @ a + b)

for i, a_i in zip(range(10), a):
    print('{}: {:.3f}'.format(i, a_i[0]))

print('Prediction: {}'.format(np.argmax(a)))
