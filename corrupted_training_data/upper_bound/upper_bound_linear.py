import logging
import numpy as np
import theano
import theano.tensor
import time
from theano.tensor.nnet import nnet
import lasagne
import lasagne.layers
from corrupted_training_data import utils
import upper_bound_iterative
from matplotlib import pyplot as plt


class LinearCorruptionOptimizer(upper_bound_iterative.CorruptionOptimizer):
    def __init__(self, C, k):
        self._k = k
        self._C = C

    def worst_corruption(self, vec_x, label_fx, w):
        assert not np.any(np.isnan(vec_x))
        values = self._C * np.abs(w) + w * vec_x * label_fx
        # indices of k largest values
        return np.argpartition(values, -self._k)[-self._k:]


class LinearUpperBoundLearner(upper_bound_iterative.UpperBoundLearner):
    """
    Inputs are limited to the [-C, C]^n box.
    """

    def __init__(self, C, regular_coef, n, gamma, use_square_loss, use_cross_entropy):
        self._regular_coef = regular_coef
        # Prepare Theano variables for inputs and targets
        # n length floating vector, 0s replace NULL entries
        self._X1_sym = theano.tensor.matrix(name='inputs_vec', dtype='float32')
        # n length binary vector, 1s for the NULL
        self._X2_sym = theano.tensor.matrix(name='inputs_binary', dtype='float32')
        self._y_sym = theano.tensor.vector(name='output', dtype='float32')

        self._w_sym = theano.shared(np.ones((n,), dtype='float64') * 0.01)

        part1 = theano.tensor.dot(self._X1_sym, self._w_sym)
        w_abs = theano.tensor.abs_(self._w_sym)
        part2 = C * theano.tensor.dot(self._X2_sym, w_abs)
        # If the adversary corrupts randomly, learner can simply treat coordinates as zeros.
        # Since data is uniform around 0, and w values are also uniform, these coordinates are
        # insignificant.
        value = - part1 * self._y_sym + part2
        self._xi = gamma - part1 * self._y_sym + part2

        p = nnet.sigmoid(value)
        self._cross_entropy_loss = self._y_sym * theano.tensor.log(p) + (self._y_sym + 1.) * theano.tensor.log(1. - p)
        self._prediction = 2 * (part1 > 0.) - 1

        if use_cross_entropy:
            # Sigmoid cross entropy loss - as in logistic regression.
            loss = -self._cross_entropy_loss
        else:
            # Margin loss - as in SVM
            loss = theano.tensor.max(self._xi, 0)
        if use_square_loss:
            loss **= 2

        self._loss = loss.mean()

        self._l2_penalty = self._regular_coef * lasagne.regularization.l2(self._w_sym)
        self._total_loss = self._loss + self._l2_penalty

        self._acc = theano.tensor.mean(theano.tensor.eq(self._prediction, self._y_sym))

        self._corruptor = utils.Corruptor()

        self.__create_functions()

    def __create_functions(self):
        self._f_loss = theano.function([self._X1_sym, self._X2_sym, self._y_sym], [self._loss])
        self._f_validation = theano.function([self._X1_sym], [self._prediction, self._l2_penalty])

    def get_hypothesis(self):
        return self._w_sym.get_value()

    def upper_bound_loss(self, corrupted_data, labels):
        zeroed_data, binary_data = self._transform_data(corrupted_data)
        return self._f_loss(zeroed_data, binary_data, labels)[0]

    def loss(self, data, labels):
        # Same implementation works for uncorrupted data
        return self.upper_bound_loss(data, labels)

    def _transform_data(self, corrupted_data):
        """
        Replace NULL coordinates with 0,
        create binary coordinates for use in the upper bound model.
        :param corrupted_data:
        :return:
        """
        result_zeroed = np.copy(corrupted_data)
        result_binary = np.zeros_like(corrupted_data)
        for i in xrange(corrupted_data.shape[0]):
            corrupted_coords = self._corruptor.corrupted_coordinates(corrupted_data[i])
            result_binary[i][corrupted_coords] = 1.
            result_zeroed[i][corrupted_coords] = 0.
        return np.float32(result_zeroed), np.float32(result_binary)

    def accuracy(self, data, labels):
        X1 = np.float32(data)
        prediction, l2_penalty = self._f_validation(X1)
        accuracy = np.average(prediction == labels)
        return accuracy

    def optimize_hypothesis(self, corrupted_data, labels, n_epochs, learning_rate,
                            validation_data, validation_labels, plot=True):
        """
        Finds the hypothesis that minimizes the naive upper bound over the loss function.
        :param corrupted_data:
        :param labels:
        :return:
        """
        self._w_sym.set_value(2 * np.random.random(self._w_sym.get_value().size) - 1.)
        sgd_updates = lasagne.updates.sgd(self._total_loss, [self._w_sym], learning_rate=learning_rate)
        f_training = theano.function([self._X1_sym, self._X2_sym, self._y_sym], outputs=[self._total_loss, self._loss],
                                     updates=sgd_updates)
        # beta = 0.8
        batch_size = 50
        logging.info('training (n_epochs, batch_size) = (' + str(n_epochs) + ', ' + str(batch_size) + ')')
        # prev_train_loss = 10000.
        i = 0
        X1, X2 = self._transform_data(corrupted_data)
        Y = np.float32(labels)
        start = time.time()
        accuracies = []
        training_losses = []
        w_size = []
        for n in xrange(n_epochs):
            for x1_batch, x2_batch, y_batch in iterate_minibatches(X1, X2, Y, batch_size, shuffle=True):
                f_training(x1_batch, x2_batch, y_batch)
            total_loss_upper_bound_training = self._f_loss(X1, X2, Y)[0]
            accuracy_validation = self.accuracy(validation_data, validation_labels)
            training_losses.append(total_loss_upper_bound_training)
            accuracies.append(accuracy_validation)
            w_size.append(np.sum(self.get_hypothesis() ** 2))
            print 'accuracy', accuracy_validation
            print '||w||', w_size[-1]
            print 'loss', total_loss_upper_bound_training
        print 'Training ({}) time elapsed: {}s'.format(n_epochs, time.time() - start)
        if plot:
            f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            ax1.plot(accuracies, '-*r')
            ax2.plot(training_losses, '-+b')
            ax3.plot(w_size, '-+g')
            plt.show()
        return accuracies, self._calculate_drop(accuracies)

    def _calculate_drop(self, values):
        for i in xrange(1, len(values)):
            if values[i] < values[i - 1] - 0.05:
                return values[i - 1]
        return values[-1]


def iterate_minibatches(inputs1, inputs2, targets, batchsize, shuffle):
    assert inputs1.shape[0] == inputs2.shape[0] == targets.shape[0]
    indices = np.arange(inputs1.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, inputs1.shape[0] - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs1[excerpt], inputs2[excerpt], targets[excerpt]
