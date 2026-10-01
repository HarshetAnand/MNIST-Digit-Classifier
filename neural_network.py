"""Neural network from scratch on two handwritten digits from MNIST.

Trains a network with one hidden layer to tell apart two digits (3 and 8 by
default). Forward propagation, backpropagation, and stochastic gradient
descent are written out by hand with NumPy only. The trained network is then
used to classify a separate file of test images, and the first test image it
gets wrong is saved for inspection.

Network: 784 input pixels -> 28 sigmoid hidden units -> 1 sigmoid output,
trained on the squared error loss.

Input files:
    mnist_train.csv  one image per row: the digit label, then 784 pixel
                     values (28x28) from 0 to 255, with no header row
    test.txt         one image per row: 784 comma-separated pixel values.
                     The first 100 images are the first digit and the rest
                     are the second digit.

Output:
    neural_network_results.txt  the learned weights and biases, the predicted
                                probability and class of each test image, and
                                the first misclassified test image
"""

import numpy as np

TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "test.txt"
RESULTS_FILE = "neural_network_results.txt"

# The two digits to tell apart. The first is class 0 and the second is class 1.
digits = [3, 8]

# Number of test images of the first digit. Images after these are the second.
num_test_class_0 = 100

# Hyperparameters.
h = 28          # hidden units
alpha = 0.01    # learning rate
num_epochs = 70


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def data_loader(file):
    """Read images and labels, scaling the pixels to the range [0, 1]."""
    a = np.genfromtxt(file, delimiter=",", skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)


x_train, y_train = data_loader(TRAIN_FILE)

# Keep only the images of the two chosen digits.
indices = np.where(np.isin(y_train, digits))[0]
x = x_train[indices]
y = y_train[indices]

# Relabel the two digits as 0 and 1.
y[y == digits[0]] = 0
y[y == digits[1]] = 1

m = x.shape[1]      # number of pixels in an image (28x28)
num_train = len(y)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(o):
    """Derivative of the sigmoid, written in terms of its output o."""
    return o * (1 - o)


def nnet(train_x, train_y, alpha, num_epochs, num_train):
    """Train the network with stochastic gradient descent.

    Each epoch visits every training image once, in random order, and updates
    the weights after each image.

    Return:
        w1 (m x h) and b1 (h x 1): hidden layer weights and biases
        w2 (h x 1) and b2 (1 x 1): output layer weights and bias
    """
    # Start from random weights and biases.
    w1 = np.random.uniform(low=-1, high=1, size=(m, h))
    w2 = np.random.uniform(low=-1, high=1, size=(h, 1))
    b1 = np.random.uniform(low=-1, high=1, size=(h, 1))
    b2 = np.random.uniform(low=-1, high=1, size=(1, 1))

    # Start with a large number, so the first loss reduction is not meaningful.
    loss_previous = 10e10

    for epoch in range(1, num_epochs + 1):
        train_index = np.arange(num_train)
        np.random.shuffle(train_index)

        for i in train_index:
            x_i = train_x[i, :]

            # Forward propagation.
            a1 = sigmoid(w1.T @ x_i.reshape(-1, 1) + b1)
            a2 = sigmoid(w2.T @ a1 + b2)

            # Backpropagation: gradient of the squared error with respect to
            # each set of weights and biases, by the chain rule.
            delta2 = (a2 - train_y[i]) * sigmoid_derivative(a2)
            delta1 = delta2 * w2 * sigmoid_derivative(a1)
            dCdw1 = delta1 * x_i.reshape(1, -1)
            dCdb1 = delta1
            dCdw2 = delta2 * a1
            dCdb2 = delta2

            # Gradient descent step.
            w1 = w1 - alpha * dCdw1.T
            b1 = b1 - alpha * dCdb1
            w2 = w2 - alpha * dCdw2
            b2 = b2 - alpha * dCdb2

        # Loss and accuracy on the full training set after each epoch.
        out_h = sigmoid(train_x @ w1 + b1.T)
        out_o = sigmoid(out_h @ w2 + b2)
        loss = 0.5 * np.sum(np.square(train_y.reshape(-1, 1) - out_o))
        loss_reduction = loss_previous - loss
        loss_previous = loss
        correct = sum((out_o > 0.5).astype(int) == train_y.reshape(-1, 1))
        accuracy = (correct / num_train)[0]
        print(
            "epoch = {:3d}".format(epoch),
            " loss = {:.7}".format(loss),
            " loss reduction = {:.7}".format(loss_reduction),
            " correctly classified = {:.4%}".format(accuracy),
        )

    return w1, b1, w2, b2


w1, b1, w2, b2 = nnet(x, y, alpha, num_epochs, num_train)


# ---------------------------------------------------------------------------
# Test predictions
# ---------------------------------------------------------------------------

x_test = np.loadtxt(TEST_FILE, delimiter=",")
x_test = x_test / 255.0

# Predicted probability of class 1 for each test image, and the class it
# implies.
a = sigmoid(x_test @ w1 + b1.T)
a = sigmoid(a @ w2 + b2)
test_probabilities = a[:, 0]
test_predictions = (test_probabilities > .5).astype(int)

# First test image of the second digit that the network labels as the first.
misclassified_image = None
for i in range(num_test_class_0, len(test_predictions)):
    if test_predictions[i] == 0:
        misclassified_image = i
        break


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

with open(RESULTS_FILE, "w") as f:
    f.write("Hidden layer weights, one row per input pixel, "
            "then one row of hidden layer biases:\n")
    w1plusb1 = np.concatenate((w1, b1.T), axis=0)
    for row in w1plusb1:
        f.write(", ".join("%.4f" % value for value in row) + "\n")

    f.write("Output layer weights, followed by the bias:\n")
    w2plusb2 = np.concatenate((w2, b2), axis=0)
    f.write(", ".join("%.4f" % value for value in w2plusb2[:, 0]) + "\n")

    f.write("Predicted probability of class 1 for each test image:\n")
    f.write(", ".join("%.2f" % value for value in test_probabilities) + "\n")

    f.write("Predicted class for each test image "
            "(0 = digit {}, 1 = digit {}):\n".format(digits[0], digits[1]))
    f.write(", ".join(str(value) for value in test_predictions) + "\n")

    if misclassified_image is None:
        f.write("No test image of digit {} was misclassified.\n".format(digits[1]))
    else:
        f.write("Pixel values of the first misclassified test image "
                "(row {}, digit {} predicted as digit {}):\n".format(
                    misclassified_image + 1, digits[1], digits[0]))
        f.write(", ".join("%.2f" % value for value in x_test[misclassified_image]) + "\n")
