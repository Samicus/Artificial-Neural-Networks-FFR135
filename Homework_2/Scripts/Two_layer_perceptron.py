import numpy as np
import pandas as pd
from numpy import genfromtxt
import math


class Layer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-2, 2, size=(out_features, in_features))
        self.bias = np.random.uniform(0, 0, size=(self.out_features))

    def forward(self, x):
        self.b = np.dot(self.weight, x) - self.bias
        return self.b


class Network:
    def __init__(self, M1, M2):
        self.learning_rate = 0.01
        self.M1 = M1
        self.M2 = M2
        self.layer_1 = Layer(2, M1)
        self.layer_2 = Layer(M1, M2)
        self.output_layer = Layer(M2, 1)
        self.layers = [self.layer_1, self.layer_2, self.output_layer]
        self.delta = np.zeros(3)
        self.V = np.zeros(3)
        self.tanh = tanh
        self.tanh_prime = tanh_prime
        self.delta_out = np.zeros(1)
        self.delta_2 = np.zeros(M2)
        self.delta_1 = np.zeros(M1)

    def forward(self, x):
        V1 = self.layer_1.forward(x)
        V1 = self.tanh(V1)
        # print("V1: ", V1)
        V2 = self.layer_2.forward(V1)
        V2 = self.tanh(V2)
        # print("V2: ", V2)
        V3 = self.output_layer.forward(V2)
        self.O = self.tanh(V3)
        # print("V3: ", V3)
        return self.O

    def compute_error(self, t):
        self.error = self.tanh_prime(self.output_layer.b) * (
            t - self.O
        )  # V3 ellr O?, b?

    def backward_propogation(self):
        self.delta_out = self.error

        self.delta_2 = (
            (self.delta_out * self.output_layer.weight)
            * self.tanh_prime(self.layer_2.b)
        ).squeeze()

        self.delta_1 = (self.delta_2 @ self.layer_2.weight) @ self.tanh_prime(
            self.layer_1.b
        )
        """
        for i in range(self.M2):  # self.M2?
            self.delta_1[i] = np.sum(
                self.delta_2[i]
                * self.layer_2.weight[i, :]
                * self.tanh_prime(self.layer_1.b[i])
            )
        """
        # print("delta_out", self.delta_out)
        # print("delta_2", self.delta_2)

    def update_weights(self, input_layer):

        self.layer_1.weight += self.learning_rate * np.outer(self.delta_1, input_layer)
        self.layer_2.weight += self.learning_rate * np.outer(
            self.delta_2, self.tanh(self.layer_1.b)
        )
        self.output_layer.weight += self.learning_rate * np.outer(
            self.delta_out, self.tanh(self.layer_2.b)
        )

    def update_bias(self):
        self.layer_1.bias -= self.learning_rate * self.delta_1
        self.layer_2.bias -= self.learning_rate * self.delta_2
        self.output_layer.bias -= self.learning_rate * self.delta_out


def signum(output):
    output[output == 0] = 1
    output = np.sign(output)
    return output


def tanh(b):
    if b.all() == 0:
        tanh = 1
    else:
        tanh = np.tanh(b)
    return tanh


def tanh_prime(b):
    if b.all() == 0:
        tanh_p = 1
    else:
        tanh_p = 1 - np.tanh(b) ** 2

    return tanh_p


if __name__ == "__main__":
    M1 = 5
    M2 = 10

    network = Network(M1, M2)
    training_set = genfromtxt("Homework_2/training_set.csv", delimiter=",")
    validation_set = genfromtxt("Homework_2/validation_set.csv", delimiter=",")
    n_epochs = 1000
    run = True

    for epoch in range(n_epochs):
        if run:
            prediction = np.zeros(len(validation_set))
            np.random.shuffle(training_set)
            for i in range(len(training_set)):
                x = training_set[i, (0, 1)]
                t = training_set[i, 2]
                network.forward(x)
                network.compute_error(t)
                network.backward_propogation()
                network.update_weights(x)
                network.update_bias()

            # validation after every 20 epochs
            # if epoch % 20 == 0:
            for j in range(len(validation_set)):
                x = validation_set[j, (0, 1)]
                t = validation_set[j, 2]
                prediction[j] = network.forward(x)

            prediction = signum(prediction)
            diff = np.abs(prediction - validation_set[:, 2])
            C = sum(diff) / (len(validation_set) * 2)
            # print(validation_set[:, 2])
            print("C:", C)
            print("epoch: ", epoch)
            if C < 0.12:  # TODO: save weights and biases
                run = False
