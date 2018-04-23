import pickle
import numpy

def save_model(weights, biases, checkpoint_filename):
    """
    Saves the current weights and biases to a checkpoint file

    param: weights - List of lists containing the network weights
    param: biases - List containing the network biases
    param: checkpoint_filename - String defining the location of the checkpoint
    """
    #Saves a list containing the weights and biases to the checkpoint file
    with open(checkpoint_filename, 'wb') as f:
        pickle.dump([weights, biases], f)
        f.close()

    print(f'Saved model at: {checkpoint_filename}')


def load_model(checkpoint_file):
    """
    Loads the saved weights and biases from a checkpoint file

    param: checkpoint_filename - String defining the location of the checkpoint
    rtype: saved_params - List containing the saved weights and biases
    """

    #Loads the weights and biases
    with open(checkpoint_file, 'rb') as f:
        saved_params = pickle.load(f)
        f.close

    print(f'Loaded weights and biases from checkpoint: {checkpoint_file}')

    return saved_params


def unzip(list):
    return zip(*list)
