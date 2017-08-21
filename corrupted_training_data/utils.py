import numpy as np


class Corruptor(object):
    def corrupt_coordinates(self, vec, coordinates, copy):
        """
        Corrupts the given ccordinates
        :param vec:
        :param coordinates:
        :return:
        """
        if copy:
            vec = np.copy(vec)
        vec[coordinates] = np.nan
        return vec

    def corrupt_randomly(self, data, k):
        result = np.copy(data)
        n = data.shape[1]
        for i in xrange(data.shape[0]):
            result[i] = self.corrupt_coordinates(data[i], self._random_indices(n, k), True)
        return result

    def _random_indices(self, n, k):
        return np.random.choice(np.arange(n), k, replace=False)

    def corrupted_coordinates(self, vec):
        return np.isnan(vec).nonzero()[0]
