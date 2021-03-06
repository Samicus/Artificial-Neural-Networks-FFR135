import numpy as np
import pickle
import matplotlib.pyplot as plt

N = 160
converged = False
pattern_to_feed = 3      # chose which distorted pattern to feed

def sign(b):        # signum function
    if b >= 0:
        return 1
    else:
        return -1


def store_pattern(p):       # Store patterns in a weight matrix
    w_i = np.zeros((N, N))
    for i in range(5):      # for every pattern
        w_i = w_i + np.outer(p[:, i], p[:, i])
    w_i = w_i / N
    np.fill_diagonal(w_i, 0)
    return w_i


def reshape(p): # reshape using a typewriter scheme
    p1 = np.reshape(p, (16, 10))
    return p1


def initialize_patterns():  # load the correct patterns from a pkl file ( 0, 1, 2, 3, 4)
    with open('../patterns.pkl', 'rb') as f1:
        patterns = pickle.load(f1)          # patterns is a 5x160 ndarray where every column is a pattern
        patterns = np.array(patterns)
    return patterns


def feed_pattern(p_nr):             # Choose which pattern to feed

    if p_nr == 1:
        x_feed = np.array([[-1, 1, 1, 1, 1, -1, 1, 1, -1, -1], [-1, 1, 1, 1, 1, -1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
                           [1, -1, -1, -1, -1, -1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, 1, 1, -1, -1],
                           [1, -1, -1, -1, -1, -1, 1, 1, -1, -1], [-1, 1, 1, 1, 1, -1, 1, 1, -1, -1], [-1, 1, 1, 1, 1, -1, 1, 1, -1, -1], [-1, 1, 1, -1, -1, 1, -1, -1, -1, -1],
                           [-1, 1, 1, -1, -1, 1, -1, -1, -1, -1], [-1, 1, 1, -1, -1, 1, -1, -1, -1, -1], [-1, 1, 1, -1, -1, 1, -1, -1, -1, -1], [-1, 1, 1, -1, -1, 1, -1, -1, -1, -1],
                           [-1, 1, 1, 1, 1, -1, 1, 1, -1, -1], [-1, 1, 1, 1, 1, -1, 1, 1, -1, -1]])



    if p_nr == 2:
        x_feed = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], [1, 1, 1, 1, 1, -1, -1, -1, 1, 1], [1, 1, 1, 1, 1, -1, -1, -1, 1, 1],
                           [1, 1, 1, 1, 1, -1, -1, -1, 1, 1], [1, 1, 1, 1, 1, -1, -1, -1, 1, 1], [1, 1, 1, 1, 1, -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
                           [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
                           [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1]])
    if p_nr == 3:
        x_feed = np.array([[-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                           [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                           [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                           [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                           [1, 1, 1, -1, -1, -1, -1, 1, 1, 1]])


    return x_feed.flatten().T

def plot(distorted_pattern, converged_pattern):
    plt.subplot(121)
    plt.imshow(distorted_pattern, cmap="gray")
    plt.subplot(122)
    plt.imshow(converged_pattern, cmap="gray")
    plt.show()

if __name__ == "__main__":
    
    p = initialize_patterns()
    w = store_pattern(p)

    S_0 = feed_pattern(pattern_to_feed)  # S_0 <-- feed distorted pattern

    while not converged:
        S_1 = S_0
        for i in range(N):
            S_1[i] = sign((1/N) * np.dot(w[i, :], S_1))

        for i in range(5):
            if (S_1 == np.array(p[:, i])).all() or (S_1 == (-np.array(p[:, i]))).all() :  # Check convergence with the patterns
                converged = True                                                # and their inverses
            else:
                S_0 = S_1

    new_state = reshape(S_1)
    distorted_pattern = reshape(feed_pattern(pattern_to_feed))
    plot(distorted_pattern, new_state)
