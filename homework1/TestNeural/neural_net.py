import numpy as np


class NeuralNetwork():

    def __init__(self):

        np.random.seed(1)

        self.synaptic_weigths = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivitive(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iteration):
        for iteration in range(training_iteration):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivitive(output))
            self.synaptic_weigths += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weigths))

        return output

