import numpy as np
from corrupted_training_data import utils


class IterativeCorruption(object):
    """
    Find iteratively, a corruption that maximizes the upper bound loss.
    """
    def __init__(self, learner, corruption_optimizer, training_data, data_labels, k):
        assert training_data.ndim == 2
        assert data_labels.ndim == 1
        assert len(training_data) == len(data_labels)
        assert isinstance(learner, UpperBoundLearner)
        assert isinstance(corruption_optimizer, CorruptionOptimizer)
        self._k = k
        self._data = training_data
        self._data_labels = np.float32(data_labels)
        self._learner = learner
        self._corruption_optimizer = corruption_optimizer
        self._corrupter = utils.Corruptor()
        self._corrupted_data = self._corrupter.corrupt_randomly(self._data, self._k)

    def iterative_loss(self, n_iterations, validation_data, validation_labels, learning_rate):
        upper_bound_losses = []
        real_losses = []
        validation_accuracies = []
        for i in xrange(n_iterations):
            self._learner.optimize_hypothesis(self._corrupted_data, self._data_labels, 20, learning_rate)
            self._corrupted_data = self._maximize_upper_bound(self._data, self._data_labels, self._learner.get_hypothesis())
            upper_bound_losses.append(self._learner.upper_bound_loss(self._corrupted_data, self._data_labels))
            real_losses.append(self._learner.loss(self._data, self._data_labels))
            validation_accuracies.append(self._learner.accuracy(validation_data, validation_labels))
        return upper_bound_losses, real_losses, validation_accuracies

    def _maximize_upper_bound(self, data, labels, hypothesis):
        res = np.copy(data)
        for i in xrange(data.shape[0]):
            indices_to_corrupt = self._corruption_optimizer.worst_corruption(data[i], labels[i], hypothesis)
            res[i] = self._corrupter.corrupt_coordinates(data[i], indices_to_corrupt, True)
        return res


class CorruptionOptimizer(object):
    def worst_corruption(self, vec, label, hypothesis):
        raise NotImplementedError


class UpperBoundLearner(object):
    def get_hypothesis(self):
        raise NotImplementedError

    def upper_bound_loss(self, corrupted_data, labels):
        raise NotImplementedError

    def loss(self, data, labels):
        raise NotImplementedError

    def optimize_hypothesis(self, corrupted_data, labels, n_epochs, learning_rate):
        """
        Finds the hypothesis that minimizes the naive upper bound over the loss function.
        :param corrupted_data:
        :param labels:
        :return:
        """
        raise NotImplementedError

    def accuracy(self, data, labels):
        raise NotImplementedError
