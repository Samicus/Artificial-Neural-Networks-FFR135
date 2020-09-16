import numpy as np
import random

homework = 1                # Change here to switch between the 2 exercises
pattern_vector = [12, 24, 48, 70, 100, 120]
error_vector = []
trials = 100000
N = 120


def generate_random_pattern(pattern_nr):            # randomly generate a matrix with -1 and 1
    random_pattern = np.random.randint(-1, 1, (N, pattern_nr))  # generate a matrix with -1 and 0 values with probability 1/2
    random_pattern[random_pattern == 0] = 1                     # if an element has the value 0: set it to 1

    return random_pattern

def sign(b):
    if b >= 0:
        return 1
    else:
        return -1


if __name__ == "__main__":


    for i in pattern_vector:
        error_counter = 0
        for j in range(trials):

            p_r = generate_random_pattern(i)
            # select random pattern and a random neuron
            p_feed = random.randint(0, i-1)
            n_update = random.randint(0, N-1)

            # store pattern in the weight matrix
            w_i = 1/N * np.dot(np.array(p_r[n_update, :]), (np.conj(p_r).T))

            if homework == 1:
                w_i[n_update] = 0   # this line of code is the only difference between the two tasks

            S_0 = sign(p_r[n_update, p_feed])
            S_1 = sign(np.dot(w_i, p_r[:, p_feed]))

            if S_0 != S_1:
                error_counter += 1

        error_percentage = float(error_counter / trials)
        error_vector.append(error_percentage)
        print(error_vector)
