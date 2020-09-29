import numpy as np 
import pandas as pd 
from numpy import genfromtxt
import math

learning_rate = 0.02
T = 10


def load_inputs():
    data = genfromtxt('Homework_2/input_data_numeric.csv', delimiter=',')
    data = np.delete(data, 0, axis = 1)
    return data


def initialize_weights():

    col = 4   # change?
    rows = 16 # change?
    w = np.random.uniform(-0.2, 0.2, size = (rows,col))  # generate a matrix with values between -1 and 1
    return w

def activation_function(b):
    return math.tanh(b)

def activation_function_prime(b):
    return 1 - math.tanh(b)**2

def initialize_thresholds(N):
    thresholds = np.random.uniform(-1, 1, size = (N))  # generate a matrix with values between -1 and 1
    return thresholds


def calculate_output(x_mu, w, theta):
    output =  np.outer(w , x_mu) - theta

    return output


def sigmoid(x):
    if x < 0:
        return -1
    else:
        return 1


def compute_error(b, target, pattern):
    k_delta = activation_function_prime(b) * (target - pattern) * (1/2)
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
    thresholds = initialize_thresholds(15)
    weights = initialize_weights()
    target =  targets()
    target_str = ['B', 'D', 'E', 'F', 'A', 'C']
    for i in range(len(target)):        
        for t1 in range(10):
            for t2 in range(T):
                mu = np.random.randint(15)
                pattern = (data[mu])
                output = calculate_output(pattern, weights[mu, :], thresholds[mu])
                pattern = activation_function(output)
                error = compute_error(output, target[i], pattern)

                """
                weights[mu, :] = weights[mu, :] + learning_rate * np.dot(error, pattern )  # pattern eller error i slutet?
                thresholds[mu] = thresholds[mu] - learning_rate * error
                output = calculate_output(pattern, weights[mu, :], thresholds[mu])
                output = np.round(output.transpose()).astype(int)
                target = np.round(target.transpose()).astype(int)

                
                if (output == target[i]).all()
                    seperable = True
                    print(target_str(i) + " is linearly seperable")
                """
                print (pattern)