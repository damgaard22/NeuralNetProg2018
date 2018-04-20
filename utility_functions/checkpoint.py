import pickle
import numpy

def save_model(weights, biases, checkpoint_filename):
    with open(checkpoint_filename, 'wb') as f:
        pickle.dump([weights, biases], f)
        f.close()

    print(f'Saved model at: {checkpoint_filename}')


def load_model(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        saved_params = pickle.load(f)
        f.close

    print(f'Loaded weights and biases from checkpoint: {checkpoint_file}')

    return saved_params


def unzip(list):
    return zip(*list)
