import pandas as pd
import numpy as np


# Part 1: Setup the data
def data_loader(file):
    df = pd.read_csv(file)
    x = (df.iloc[:, 1:] / 255.0).to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return (x, y)


# load the training data
x_train, y_train = data_loader("mnist_train.csv")


# test_labels might be different for you
# 8 (label it 0) and 4 (label it 1)
test_labels = [3, 8]
indices = np.where(np.isin(y_train, test_labels))[0]

# get the indices of the training data that have labels 8 and 4
x = x_train[indices]
y = y_train[indices]

# label 8 as 0 and label 4 as 1
y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1

# Part 2: Configure the Hyperparameters

# adjust number of epochs and learning rate by yourself
num_epochs = 200
alpha = 0.01

# total number of pixels in an image (28x28)
m = x.shape[1]

# random weights and bias in the beginning
w = np.random.rand(m)
b = np.random.rand()

# Part 3: Training Logistic Regression

# start with a large number
loss_previous = 10e10

# step over the epochs
for epoch in range(num_epochs):
    # calculate the activation
    a = x @ w + b
    a = 1 / (1 + np.exp(-a))

    # bound items in a to avoid log(0):
    a = np.clip(a, 0.001, 0.999)

    # calculate the weights and bias
    w -= alpha * (x.T) @ (a - y)
    b -= alpha * (a - y).sum()

    # calculate the loss
    loss = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    loss_reduction = loss_previous - loss
    loss_previous = loss

    # calculate the accuracy
    # correct predictions / total number of predictions
    accuracy = sum((a > 0.5).astype(int) == y) / len(y)

    print(
        "epoch = ",
        epoch,
        " loss = {:.7}".format(loss),
        " loss reduction = {:.7}".format(loss_reduction),
        " correctly classified = {:.4%}".format(accuracy),
    )

# Part 4: Write
file = "result5.txt"
import os
from pathlib import Path

my_file = Path(file)
if my_file.is_file():
    os.remove(file)


# Q1: first_tranining image  the feature vector 
first = x[0]

f = open(file, "a")
f.write("##1: \n")

for i, value in enumerate(first):
    if i != 0:
        f.write(", ")
    f.write("%.2f" % value)
f.write("\n")
f.close()

# Q2: write the weights and bias
f = open(file, "a")
f.write("##2: \n")
for i, value in enumerate(w):
    if i != 0:
        f.write(", ")
    f.write("%.4f" % value)
f.write(", ")
# write the bias
f.write("%.4f" % b)
f.write("\n")
f.close()

# Q3

# Load the test 
x = np.loadtxt("sreyatest.txt", delimiter=",")
x = x / 255.0

# calculate the activation
a = 1 / (1 + np.exp(-(x @ w + b)))

f = open(file, "a")
f.write("##3: \n")
for i, value in enumerate(a):
    if i != 0:
        f.write(", ")
    f.write("%.2f" % value)
f.write("\n")
f.close()


# Q4

f = open(file, "a")
f.write("##4: \n")

# enumerate over the activations computed in Q3 (activations of the test data)
for i, value in enumerate(a):
    if value >= 0.5:
        value = 1
    else:
        value = 0
    if i != 0:
        f.write(", ")
    f.write(str(value))
f.write("\n")
f.close()
