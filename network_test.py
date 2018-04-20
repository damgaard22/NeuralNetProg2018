import mnist_data
from network import Network

print(f'Loading data... ')

training_data, validation_data, test_data = mnist_data.load_data_wrapper()

print(f'Constructing network... ')

# net = Network([784, 2, 10], checkpoint_file='./model_checkpoint')
net = Network([784, 30, 10], checkpoint_file='./model_checkpoint')

# net.SGD(training_data, 30, 10, 3.0, test_data=test_data, checkpoint_file='./model_checkpoint', checkpoint_model_best=True)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)