import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivitive(x):
    return x * (x - 1)


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(100000):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = outputs - training_outputs

    adjustments = error * sigmoid_derivitive(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print('synaptic weights after training: ')
print(synaptic_weights)

print('Outputs after traning 1 iteration: ')
for a in outputs:
    print ("%.10f" % a)
print(outputs)
