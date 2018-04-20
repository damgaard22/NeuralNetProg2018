import numpy as np

def loss_function(output_activations, y):
        return (output_activations - y)  

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
