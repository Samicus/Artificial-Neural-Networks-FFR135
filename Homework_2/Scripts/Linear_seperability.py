import numpy as np
import pandas as pd
from numpy import genfromtxt
import math

learning_rate = 0.02
T = 10 ** 5


def load_inputs():
    data = genfromtxt("Homework_2/input_data_numeric.csv", delimiter=",")
    data = np.delete(data, 0, axis=1)
    return np.array(data)


def initialize_weights():

    w = np.random.uniform(
        -0.2, 0.2, size=(4)
    )  # generate a matrix with values between -0.2 and 0.2
    return np.asmatrix(w)


def activation_function(b):
    return np.tanh(b)


def activation_function_prime(b):
    return 1 - np.tanh(b) ** 2


def initialize_thresholds():
    thresholds = np.random.uniform(
        -1, 1
    )  # generate a matrix with values between -1 and 1
    return thresholds


def calculate_output(x_mu, w, theta):
    b = np.dot(w, x_mu) - theta
    output = activation_function((1 / 2) * b)

    return output, b


def signum(output):
    output_copy = np.copy(output)
    output_copy[output_copy == 0] = 1
    output_copy = np.sign(output_copy)
    output_copy = np.round(output_copy.transpose()).astype(int)
    return output_copy


def compute_error(b, target, output):
    k_delta = activation_function_prime(b) * (target - output)
    return k_delta


def targets():
    B = [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1]
    D = [1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1]
    E = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1]
    F = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1]
    A = [1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1]
    C = [-1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1]

    targets = [B, D, E, F, A, C]

    return np.array(targets)


if __name__ == "__main__":

    data = load_inputs()
    thresholds = initialize_thresholds()
    weights = initialize_weights()
    target = np.asmatrix(targets())
    target_str = ["B", "D", "E", "F", "A", "C"]

    for i in range(len(target)):
        seperable = False
        output = np.zeros(16)
        for t1 in range(10):
            thresholds = initialize_thresholds()
            weights = initialize_weights()
            for t2 in range(T):
                if seperable == False:
                    mu = np.random.randint(16)  # choose random mu
                    pattern = data[mu]
                    output[mu], b = calculate_output(
                        pattern, weights, thresholds  # calculate output
                    )
                    error = compute_error(
                        b, target[i, mu], output[mu]
                    )  # Compute the error

                    weights = weights + learning_rate * error * pattern  # update weight
                    thresholds = thresholds - learning_rate * error  # update threshold

                    if (
                        signum(output) == target[i]
                    ).all():  # check for linear seperability
                        seperable = True
                        print(target_str[i] + " is linearly seperable")
