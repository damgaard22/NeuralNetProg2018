import numpy as np

def loss_function(output_activations, y):
    """
    The networks loss function. ME - Minimum Error
    """
    return (output_activations - y)  

def sigmoid(z):
    """
    Numpy implementation of the sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))
