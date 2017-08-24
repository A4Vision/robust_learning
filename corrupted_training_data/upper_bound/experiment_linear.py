import cPickle
import gzip
import numpy as np
import os
from corrupted_training_data.upper_bound import upper_bound_iterative, upper_bound_linear


class Problem(object):
    def __init__(self, k, C):
        self.k = k
        self.C = C


def create_data(C, k, n, amount):
    data = (2 * np.random.random(size=(amount, n)) - 1.) * C
    w = np.random.random(size=(n,)) * 2 - 1.
    values = np.dot(data, w)
    indices_of_good_examples = (np.abs(values) > C * k / 5.).nonzero()
    data = data[indices_of_good_examples]
    values = values[indices_of_good_examples]
    print np.average(np.abs(values)), np.var(np.abs(values))
    labels = np.sign(values)
    return data, labels


def experiment1():
    np.random.seed(123)
    learning_rate = 0.01
    regular_coef = 0.
    C = 1.
    k = 5
    n_iterations = 10
    n = 20
    data_amount = 10000
    data, labels = create_data(C, k, n, data_amount)
    print len(data)
    first = int(data.shape[0] * 0.9)
    train_data, train_labels = data[:first], labels[:first]
    validation_data, validation_labels = data[first:], labels[first:]

    adversary = upper_bound_linear.LinearCorruptionOptimizer(C, k)
    learner = upper_bound_linear.LinearUpperBoundLearner(C, regular_coef, n, 2., False, True)
    iterative_adversary = upper_bound_iterative.IterativeCorruption(learner, adversary, train_data, train_labels, k)
    iterative_adversary.iterative_loss(n_iterations, validation_data, validation_labels, learning_rate)


def mnist():
    with gzip.open(os.path.expanduser("~/data/mnist.pkl.gz")) as f:
        train, valid, test = cPickle.load(f)
        return train, valid, test


def multilabel_to_binary(data, labels, label0, label1):
    indices0 = (labels == label0)
    indices1 = (labels == label1)
    new_data = data[indices0 | indices1]
    new_labels = np.array([2 * int(label == label1) - 1 for label in labels if label in (label0, label1)])
    return new_data, new_labels


def normalize_data_to_cube(data, C):
    max_, min_ = np.min(data), np.max(data)
    gap = max_ - min_
    zeroed = (data - min_)
    centered = zeroed - gap / 2.
    normalized = centered * C / (gap / 2.)
    assert abs(np.max(normalized) - C) < 1e-5
    assert abs(np.min(normalized) + C) < 1e-5
    return normalized


def experiment_mnist():
    np.random.seed(123)
    learning_rate = 0.1
    regular_coef = 0.01
    C = 1.
    k = 20
    n_iterations = 4
    n = 784
    gamma = .1

    train, valid, test = mnist()
    train_non_normalized, train_labels = multilabel_to_binary(train[0], train[1], 3, 8)
    valid_non_normalized, valid_labels = multilabel_to_binary(valid[0], valid[1], 3, 8)
    train_data = normalize_data_to_cube(train_non_normalized, C)
    valid_data = normalize_data_to_cube(valid_non_normalized, C)

    adversary = upper_bound_linear.LinearCorruptionOptimizer(C, k)
    learner = upper_bound_linear.LinearUpperBoundLearner(C, regular_coef, n, gamma, True, False)
    iterative_adversary = upper_bound_iterative.IterativeCorruption(learner, adversary, train_data, train_labels, k)
    iterative_adversary.iterative_loss(n_iterations, valid_data, valid_labels, learning_rate)


def main():
    # experiment1()
    experiment_mnist()


if __name__ == '__main__':
    main()