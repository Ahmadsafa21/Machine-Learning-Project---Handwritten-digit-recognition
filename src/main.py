import mnist_loader
import network
#run this code from src library, ---> cd src
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
test_data = list(test_data)
training_data = list(training_data)
validation_data = list(training_data)
print(len(test_data))
print(type(training_data) )
net = network.Network([784,30, 10])
#Un-optimized hyper parameters: 784, 0, 10
net.SGD(training_data, 30, 10, 3, test_data)


