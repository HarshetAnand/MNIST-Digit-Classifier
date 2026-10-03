# Handwritten Digit Classification - Logistic Regression and Neural Networks

A from-scratch implementation of logistic regression and a single-hidden-layer neural network to classify handwritten digits from the MNIST dataset. Built without ML libraries to demonstrate understanding of the underlying mathematics including forward propagation, backpropagation, and gradient descent.

## Features

- Binary classification of handwritten digit images (pixel intensities as inputs)
- Logistic regression model with gradient descent optimization
- Single hidden layer neural network with sigmoid activation
- Custom forward propagation and backpropagation implementation
- Loss tracking and accuracy reporting per epoch
- Test prediction and misclassification analysis

## Tech Stack

- Python
- NumPy (numerical computation)
- Pandas (data loading)
- MNIST dataset

## Data

The scripts expect two files in the project folder:

- `mnist_train.csv`: the MNIST training set in CSV format, one image per row (the digit label, then 784 pixel values from 0 to 255).
- `test.txt`: unlabeled test images, one per row, as 784 comma-separated pixel values. `neural_network.py` assumes the first 100 rows are the first digit and the rest are the second.

Both scripts classify the digits 3 and 8 by default. Change the `digits` list at the top of either file to use a different pair.

## Implementation Details

**Logistic Regression (logistic_regression.py)**

- 200 epochs, learning rate 0.01
- Sigmoid activation with cross-entropy loss
- Manual weight and bias updates via gradient descent

**Neural Network (neural_network.py)**

- 1 hidden layer with 28 units
- 70 epochs, learning rate 0.01
- Sigmoid activations throughout
- Backpropagation implemented from scratch

## Key Concepts Demonstrated

- Building ML models from scratch without high-level libraries
- Forward propagation and backpropagation mathematics
- Gradient descent optimization
- Sigmoid activation and its derivative
- Loss functions and accuracy metrics
- MNIST image preprocessing and normalization
