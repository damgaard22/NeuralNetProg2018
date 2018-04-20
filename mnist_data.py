import pickle
import gzip

import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    return (training_data, validation_data, test_data)


def load_data_wrapper():
    train_data, val_data, testing_data = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    training_results = [vector_result(y) for y in train_data[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    validation_data = zip (validation_inputs, val_data[1])

    test_inputs = [np.reshape(x, (784,1)) for x in testing_data[0]]
    test_data = zip(test_inputs, testing_data[1])

    return(list(training_data), list(validation_data), list(test_data))


def vector_result(j):
    result = np.zeros((10, 1))
    result[j] = 1.0
    
    return result