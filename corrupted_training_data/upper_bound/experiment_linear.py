import numpy as np
from corrupted_training_data.upper_bound import upper_bound_iterative, upper_bound_linear


class Problem(object):
    def __init__(self, k, C):
        self.k = k
        self.C = C


def create_data(C, k, n, amount):
    data = (2 * np.random.random(size=(amount, n)) - 1.) * C
    w = np.random.random(size=(n,)) * 2 - 1.
    values = np.dot(data, w)
    print values
    print np.average(values), np.var(values)
    labels = np.sign(values)
    return data, labels


def experiment1():
    learning_rate = 0.00001
    regular_coef = 0.001
    C = 1.
    k = 5
    n_iterations = 10
    n = 20
    data_amount = 10000
    data, labels = create_data(C, k, n, data_amount)
    first = int(data.shape[0] * 0.9)
    train_data, train_labels = data[:first], labels[:first]
    validation_data, validation_labels = data[first:], labels[first:]

    adversary = upper_bound_linear.LinearCorruptionOptimizer(C, k)
    learner = upper_bound_linear.LinearUpperBoundLearner(C, regular_coef, n, True)
    iterative_adversary = upper_bound_iterative.IterativeCorruption(learner, adversary, train_data, train_labels, k)
    iterative_adversary.iterative_loss(n_iterations, validation_data, validation_labels, learning_rate)


def main():
    experiment1()


if __name__ == '__main__':
    main()