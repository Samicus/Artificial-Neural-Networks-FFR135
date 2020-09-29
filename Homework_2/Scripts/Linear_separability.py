import numpy as np
import pandas as pd
from numpy import genfromtxt
import math

learning_rate = 0.02
T = 10 ** 5


def load_inputs():
    data = genfromtxt("Homework_2/input_data_numeric.csv", delimiter=",")
    data = np.delete(data, 0, axis=1)
    return data


def initialize_weights():

    col = 4  # change?
    rows = 16  # change?
    w = np.random.uniform(
        -0.2, 0.2, size=(rows, col)
    )  # generate a matrix with values between -1 and 1
    return w


def activation_function(b):
    return math.tanh(b)


def activation_function_prime(b):
    return 1 - math.tanh(b) ** 2


def initialize_thresholds(N):
    thresholds = np.random.uniform(
        -1, 1, size=(N)
    )  # generate a matrix with values between -1 and 1
    return thresholds


def calculate_output(x_mu, w, theta):
    b = np.dot(w, x_mu) - theta
    output = activation_function((1 / 2) * b)

    return output, b


def sigmoid(output):
    output_copy = np.copy(output)
    output_copy[output_copy == 0] = 1
    output_copy = np.sign(output_copy)
    output_copy = np.round(output_copy.transpose()).astype(int)
    return output_copy


def compute_error(b, target, pattern):
    k_delta = activation_function_prime(b) * (target - pattern)
    return k_delta


def targets():
    B = np.array([1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1]).T
    D = np.array([1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1]).T
    E = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1]).T
    F = np.array([-1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1]).T
    A = np.array([1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1]).T
    C = np.array([-1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1]).T

    return np.asmatrix([B, D, E, F, A, C])


if __name__ == "__main__":

    data = load_inputs()
    thresholds = initialize_thresholds(16)
    weights = initialize_weights()
    target = targets()
    target_str = ["B", "D", "E", "F", "A", "C"]

    for i in range(len(target)):
        seperable = False
        output = np.zeros(16)
        for t1 in range(10):
            thresholds = initialize_thresholds(16)
            weights = initialize_weights()
            for t2 in range(T):
                if seperable == False:
                    mu = np.random.randint(16)
                    pattern = data[mu]
                    output[mu], b = calculate_output(
                        pattern, weights[mu, :], thresholds[mu]
                    )
                    error = compute_error(b, target[i, mu], output[mu])

                    weights[mu, :] = weights[mu, :] + learning_rate * error * output[mu]
                    thresholds[mu] = thresholds[mu] - learning_rate * error

                    if (sigmoid(output) == target[i]).all():
                        seperable = True
                        print(target_str[i] + " is linearly seperable")
                        print(sigmoid(output))
                        print(target[i])
