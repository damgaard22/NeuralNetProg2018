import pickle
import numpy


def save_model(weights, biases, checkpoint_filename):
    with open(checkpoint_filename, 'wb') as f:
        pickle.dump(list(zip(weights, biases)), f)
        f.close()

    print(f'Saved model at: {checkpoint_filename}')

def load_model(checkpoint):
    with open(checkpoint, 'rb') as f:
        saved_params = pickle.load(f)
        f.close

    return unzip(saved_params)


def unzip(list):
    return zip(*list)