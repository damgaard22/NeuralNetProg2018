import numpy as np
import random
import utility_functions.math as math
import utility_functions.checkpoint as checkpoint
import pickle

class Network(object):
    
    def __init__(self, sizes, checkpoint_file=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.checkpoint = checkpoint_file
        self.best_acc = 0

        if self.checkpoint:
            assert isinstance(self.checkpoint, str), 'Must provide the checkpoint file in order to load the model'
            
            params = checkpoint.load_model(self.checkpoint)

            self.weights = [w for w,b in params]
            self.biases = [b for w,b in params]

        else:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    
    def feedforward(self, activation):
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
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.prop_mini_batch(mini_batch, eta)

            if checkpoint_model_best or checkpoint_model_epoch:
                assert isinstance(checkpoint_file, str), 'Must provide checkpoint file in order to save the model' 
                
            
            if test_data:
                result = self.evaluate(test_data)
                accuracy = result / n_test * 100
                print(f'Epoch {i}: {result} / {n_test} || Acc: {accuracy}%')
                
                if checkpoint_model_best and checkpoint_file and accuracy > self.best_acc:
                    checkpoint.save_model(self.weights, self.biases, checkpoint_file)
                    self.best_acc = accuracy

            else: 
                print(f'Epoch {i} done')

            if checkpoint_model_epoch and checkpoint_file:
                if i % checkpoint_model_epoch == 0:
                    checkpoint.save_model(self.weights, self.biases, checkpoint_file)

    
    def prop_mini_batch(self, mini_batch, eta):
        diff_b = [np.zeros(bias.shape) for bias in self.biases]
        diff_w = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in mini_batch:
            delta_diff_b, delta_diff_w = self.backpropagation(x, y)
            diff_b = [db + ddb for db, ddb in zip(diff_b, delta_diff_b)]
            diff_w = [dw + ddw for dw, ddw in zip(diff_w, delta_diff_w)]

        self.weights = [w - (eta / len(mini_batch)) * dw for w, dw in zip(self.weights, diff_w)]
        self.biases = [b - (eta / len(mini_batch)) * db for b, db in zip(self.biases, diff_b)]


    def backpropagation(self, x, y):
        diff_b = [np.zeros(bias.shape) for bias in self.biases]
        diff_w = [np.zeros(weight.shape) for weight in self.weights]

        activation = x 
        activations = [x]
        z_list = []

        print(self.weights)  
        print(self.biases)
        print(activations)

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_list.append(z)
            activation = math.sigmoid(z)
            activations.append(activation)
        
        delta = math.loss_function(activations[-1], y) * math.sigmoid_prime(z_list[-1])
        diff_b[-1] = delta
        diff_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = z_list[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * math.sigmoid_prime(z)
            diff_b[-l] = delta
            diff_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (diff_b, diff_w)


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)




