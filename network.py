import numpy as np
import random
import utility_functions.math as math
import utility_functions.checkpoint as checkpoint
import pickle

class Network(object):
    """
    Class defining an Artificial Neural Network (ANN)
    """
    
    def __init__(self, sizes, checkpoint_file=None):
        """
        Initializes the network.
        Defines the network layout and initial weights and biases

        param: sizes - Array of layer sizes.
        param: checkpoint_file - String defining checkpoint file to load weights and biases from
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.checkpoint = checkpoint_file
        self.best_acc = 0

        #Checks if user defined Network with a checkpoint
        if self.checkpoint:
            #Assures that the checkpoint variable is a string
            assert isinstance(self.checkpoint, str), 'Must provide the checkpoint file in order to load the model'
            
            #Loads the checkpoint configuration and defines the weights and biases.
            params = checkpoint.load_model(self.checkpoint)

            self.weights = [w for w,b in params]
            self.biases = [b for w,b in params]

        else:
            #If no checkpoint is defined, weights and biases are initalized randomly
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    
    def feedforward(self, activation):
        """
        Computes the feedforward pass and returns the calculated activations

        param: sizes - Array of layer sizes.
        param: checkpoint_file - String defining checkpoint file to load weights and biases from
        rtype: activation - Array of activations
        """
        for bias, weight in zip(self.biases, self.weights):
            activation = math.sigmoid(np.dot(weight, activation) + bias)
      
        return activation


    def SGD(
            self, 
            training_data, 
            epochs, 
            mini_batch_size, 
            eta, 
            test_data=None, 
            checkpoint_model_epoch=None,
            checkpoint_model_best=False,
            checkpoint_file=None,
        ):
        """
        Performs Stochastic Gradient Descent in order to minimize the error (loss_function())

        param: training_data - Iterable in the form of (input, result)
        param: epochs - Integer defining the amount of times to train
        param: mini_batch_size - The size of the mini batches
        param: eta - Float defining the learning rate of the SGD algorithm
        param: test_data - Iterable in the form of (input, result)
        param: checkpoint_model_epoch - Integer defining the interval at which the model should be saved
        param: checkpoint_model_best - Boolean flag activating model checkpoints every accuracy improvement
        param: checkpoint_file - String defining the location of the checkpoint
        """
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        
        #Trains for specified number of epochs
        for i in range(epochs):
            
            #Shuffles the training the avoid overfitting on data structure
            random.shuffle(training_data)

            #Seperates the training_data into batches 
            mini_batches = [
                training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            #Performs backpropagation for every mini batch
            for mini_batch in mini_batches:
                self.prop_mini_batch(mini_batch, eta)

            #Checks if the checkpoint file is defined correctly
            if checkpoint_model_best or checkpoint_model_epoch:
                assert isinstance(checkpoint_file, str), 'Must provide checkpoint file in order to save the model' 
                
            #If in testing mode log the training results
            if test_data:
                result = self.evaluate(test_data)
                accuracy = result / n_test * 100
                print(f'Epoch {i}: {result} / {n_test} || Acc: {accuracy}%')
                
                #Saves the model if it improved over the current best accuracy
                if checkpoint_model_best and checkpoint_file and accuracy > self.best_acc:
                    checkpoint.save_model(self.weights, self.biases, checkpoint_file)
                    self.best_acc = accuracy
            
            else: 
                print(f'Epoch {i} done')
            
            #Saves the model at every specified number of epochs
            if checkpoint_model_epoch and checkpoint_file:
                if i % checkpoint_model_epoch == 0:
                    checkpoint.save_model(self.weights, self.biases, checkpoint_file)

    
    def prop_mini_batch(self, mini_batch, eta):
        """
        Updates the weights and biases to minimize the loss function

        param: mini_batch - List of training data tuples
        param: eta - Float defining the learning rate
        """

        #Creates empty Numpy arrays to store the weights and biases derivatives 
        diff_b = [np.zeros(bias.shape) for bias in self.biases]
        diff_w = [np.zeros(weight.shape) for weight in self.weights]

        
        for x, y in mini_batch:
            #Performs backpropagation to calculate the derivative wrt. the loss function
            delta_diff_b, delta_diff_w = self.backpropagation(x, y)


            diff_b = [db + ddb for db, ddb in zip(diff_b, delta_diff_b)]
            diff_w = [dw + ddw for dw, ddw in zip(diff_w, delta_diff_w)]

        #Updates the weights and biases
        self.weights = [w - (eta / len(mini_batch)) * dw for w, dw in zip(self.weights, diff_w)]
        self.biases = [b - (eta / len(mini_batch)) * db for b, db in zip(self.biases, diff_b)]


    def backpropagation(self, x, y):
        """
        Performs backpropagation on training data

        param: x - Input from training data
        param: y - Expected output from training data
        rtype: diff_b, diff_b - Derivative of the weights and biases wrt. loss
        """
        
        #Creates empty Numpy arrays to store the weights and biases derivatives 
        diff_b = [np.zeros(bias.shape) for bias in self.biases]
        diff_w = [np.zeros(weight.shape) for weight in self.weights]

        #Creates lists to store intermediate values and activations
        activation = x 
        activations = [x]
        z_list = []

        #Calculates activations and add the to the activation list
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_list.append(z)
            activation = math.sigmoid(z)
            activations.append(activation)
        
        #Calculates the loss differential
        delta = math.loss_function(activations[-1], y) * math.sigmoid_prime(z_list[-1])

        #Calcuates the derivative of the weights and biases in the last layer wrt. the loss function
        diff_b[-1] = delta
        diff_w[-1] = np.dot(delta, activations[-2].transpose())

        #Backwards pass through the layes calculating the derivative of the weights and biases wrt. loss
        for l in range(2, self.num_layers):
            z = z_list[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * math.sigmoid_prime(z)
            diff_b[-l] = delta
            diff_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (diff_b, diff_w)


    def evaluate(self, test_data):
        """
        Evaluates the networks output wrt. the expected output
        Sums the answers.

        param: test_data - Tuple of input and expected results
        rtype: - Amount of correct answers made by the network
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)




