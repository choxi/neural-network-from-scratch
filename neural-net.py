import  numpy as np
import  matplotlib.pyplot as plt

import  sklearn
from    sklearn import datasets, linear_model

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)

# Variables
num_examples    = len(X)    # training set size
nn_input_dim    = 2         # input layer dimensionality
nn_output_dim   = 2         # output layer dimensionality

epsilon         = 0.01      # learning rate for gradient descent
reg_lambda      = 0.01      # regularization strength

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propogation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    output = softmax(z2)

    # Calculate loss
    # TODO: what is "probs"?
    correct_logprobs = -np.log(probs[range(num_examples)])
    data_loss        = np.sum(correct_logprobs)

    # Add regularization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1./num_examples * data_loss

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propogation
    z1      = x.dot(W1) + b1
    a1      = np.tanh(z1)
    z2      = a1.dot(W2) + b2
    output  = softmax(z2)

    return np.argmax(output, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    # Initialize weights and biases to random values
    # TODO: why do we divide the weights by the square root of the input size?
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    for i in xrange(0, num_passes):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        probs = softmax(z2)

        # Backpropogation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        model = { 'W1': W1, 'b1': b1, 'W2', W2, 'b2': b2 }

        # Print the loss. Expensive because it uses the whole dataset, so we only print it once every 1000 iterations.
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(model))

    return model

def softmax(vector):
    exponent_vector = np.exp(vector)
    probabilities   = exponent_vector / np.sum(exponent_vector, axis=1, keepdims=True)
