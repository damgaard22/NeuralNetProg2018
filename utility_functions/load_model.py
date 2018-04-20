import pickle

def load_model(checkpoint):
    with open(checkpoint, 'rb') as f:
        saved_params = pickle.load(f)
        f.close

    return unzip(saved_params)


def unzip(list):
    return zip(*list)