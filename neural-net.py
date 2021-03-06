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

    probabilities       = forward_propogate(model, X)
    correct_logprobs    = -np.log(probabilities[range(num_examples)])
    data_loss           = np.sum(correct_logprobs)

    # Add regularization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1./num_examples * data_loss

def forward_propogate(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    probabilities  = softmax(z2)

    return probabilities

def predict(model, x):
    probabilities = forward_propogate(model, x)
    return np.argmax(probabilities, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    # Initialize weights and biases to random values
    np.random.seed(0)

    model = {
      'W1': np.random.randn(nn_input_dim, nn_hdim),
      'b1': np.zeros((1, nn_hdim)),
      'W2': np.random.randn(nn_hdim, nn_output_dim),
      'b2': np.zeros((1, nn_output_dim))
    }

    for i in xrange(0, num_passes):
        # Forward propagation
        z1 = X.dot(model['W1']) + model['b1']
        a1 = np.tanh(z1)
        z2 = a1.dot(model['W2']) + model['b2']
        probs = softmax(z2)

        # Backpropogation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(model['W2'].T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms
        dW2 += reg_lambda * model['W2']
        dW1 += reg_lambda * model['W1']

        model['W1'] += -epsilon * dW1
        model['b1'] += -epsilon * db1
        model['W2'] += -epsilon * dW2
        model['b2'] += -epsilon * db2

        # Print the loss. Expensive because it uses the whole dataset, so we only print it once every 1000 iterations.
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(model))

    return model

def softmax(vector):
    exponent_vector = np.exp(vector)
    probabilities   = exponent_vector / np.sum(exponent_vector, axis=1, keepdims=True)

    return probabilities

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Run the model
model = build_model(3, print_loss=True)
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()
