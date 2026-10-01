"""Logistic regression from scratch on two handwritten digits from MNIST.

Trains a binary classifier to tell apart two digits (3 and 8 by default)
using batch gradient descent on the cross-entropy loss, with NumPy only.
The trained model is then used to classify a separate file of test images.

Input files:
    mnist_train.csv  one image per row: the digit label, then 784 pixel
                     values (28x28) from 0 to 255
    test.txt         one image per row: 784 comma-separated pixel values

Output:
    logistic_regression_results.txt  the first training image, the learned
                                     weights and bias, and the predicted
                                     probability and class of each test image
"""

import numpy as np
import pandas as pd

TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "test.txt"
RESULTS_FILE = "logistic_regression_results.txt"

# The two digits to tell apart. The first is class 0 and the second is class 1.
digits = [3, 8]

# Hyperparameters.
num_epochs = 200
alpha = 0.01  # learning rate


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def data_loader(file):
    """Read images and labels, scaling the pixels to the range [0, 1]."""
    df = pd.read_csv(file)
    x = (df.iloc[:, 1:] / 255.0).to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return (x, y)


x_train, y_train = data_loader(TRAIN_FILE)

# Keep only the images of the two chosen digits.
indices = np.where(np.isin(y_train, digits))[0]
x = x_train[indices]
y = y_train[indices]

# Relabel the two digits as 0 and 1.
y[y == digits[0]] = 0
y[y == digits[1]] = 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Number of pixels in an image (28x28).
m = x.shape[1]

# Start from random weights and bias.
w = np.random.rand(m)
b = np.random.rand()

# Start with a large number, so the first loss reduction is not meaningful.
loss_previous = 10e10

for epoch in range(num_epochs):
    # Predicted probability of class 1 for every image, bounded away from 0
    # and 1 to avoid log(0) in the loss.
    a = sigmoid(x @ w + b)
    a = np.clip(a, 0.001, 0.999)

    # Gradient descent step on the cross-entropy loss.
    w -= alpha * (x.T) @ (a - y)
    b -= alpha * (a - y).sum()

    loss = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    loss_reduction = loss_previous - loss
    loss_previous = loss

    # Share of training images classified correctly.
    accuracy = sum((a > 0.5).astype(int) == y) / len(y)

    print(
        "epoch = {:3d}".format(epoch),
        " loss = {:.7}".format(loss),
        " loss reduction = {:.7}".format(loss_reduction),
        " correctly classified = {:.4%}".format(accuracy),
    )


# ---------------------------------------------------------------------------
# Test predictions
# ---------------------------------------------------------------------------

x_test = np.loadtxt(TEST_FILE, delimiter=",")
x_test = x_test / 255.0

# Predicted probability of class 1 for each test image, and the class it
# implies.
test_probabilities = sigmoid(x_test @ w + b)
test_predictions = [1 if p >= 0.5 else 0 for p in test_probabilities]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

with open(RESULTS_FILE, "w") as f:
    f.write("Feature vector of the first training image:\n")
    f.write(", ".join("%.2f" % value for value in x[0]) + "\n")

    f.write("Learned weights, followed by the bias:\n")
    f.write(", ".join("%.4f" % value for value in w))
    f.write(", " + "%.4f" % b + "\n")

    f.write("Predicted probability of class 1 for each test image:\n")
    f.write(", ".join("%.2f" % value for value in test_probabilities) + "\n")

    f.write("Predicted class for each test image "
            "(0 = digit {}, 1 = digit {}):\n".format(digits[0], digits[1]))
    f.write(", ".join(str(value) for value in test_predictions) + "\n")
