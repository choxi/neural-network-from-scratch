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
    correct_logprobs = -np.log(probs[range(num_examples)])

def softmax(vector):
    exponent_vector = np.exp(vector)
    probabilities   = exponent_vector / np.sum(exponent_vector, axis=1, keepdims=True)
