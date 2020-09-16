import numpy as np
import random
import math

beta = 2
nr_of_patterns = 7
N = 200
T = 2*10**5


def stochastic_async_update(b):     # update state
    rand = random.random()
    p = probabilty_function(b)
    if p > rand:
        return 1
    else:
        return -1


def probabilty_function(b):
    p = 1 / (1 + math.exp(-2*beta*b))
    return p


def generate_random_pattern(nr_of_patterns):            # randomly generate a matrix with -1 and 1
    random_pattern = np.random.randint(-1, 1, (N, nr_of_patterns))  # generate a matrix with -1 and 0 values with probability 1/2
    random_pattern[random_pattern == 0] = 1                     # if an element has the value 0: set it to 1

    return random_pattern


def store_pattern(p):       # Store patterns in a weight matrix
    w_i = np.zeros((N, N))
    for i in range(len(p[0,:])):      # for every pattern
        w_i = w_i + np.outer(p[:, i], p[:, i])
    w_i = w_i / N
    np.fill_diagonal(w_i, 0)
    return w_i


if __name__ == "__main__":

    m_array = np.zeros(100)
    for i in range(100):
        temp_mu = np.zeros(T)

        p_r = generate_random_pattern(nr_of_patterns)  # generate random patterns
        w_i = store_pattern(p_r)  # store random patterns
        S_0 = np.copy(p_r[:, 0])  # feed x(1)

        for j in range(T):
            S_1 = S_0
            n_update = random.randint(0, N-1)   # choose random neuron to update
            b = np.dot(w_i[n_update, :], np.transpose(S_0))
            S_1[n_update] = stochastic_async_update(b)  #update asynchronously
            temp_mu[j] = np.dot(S_1, np.array(p_r[:, 0])) / N
            S_0 = S_1

        m_array[i] = np.sum(temp_mu) / T
        print(m_array)

    m_average = np.sum(m_array) / 100
    print(m_average)