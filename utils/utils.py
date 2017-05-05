from scipy import special
import math
import numpy
import collections


def log2_prob_binom(n, k, p):
    """

    :param n:
    :param k:
    :param p:
    :return:
    """
    return math.log(special.binom(n, k), 2) + k * math.log(p, 2) + (n - k) * math.log(1 - p, 2)


def log2_subtract(log2_x, log2_y):
    assert log2_x > log2_y
    return log2_x + math.log1p(-2 ** (log2_y - log2_x)) / math.log(2)



def main():
    for a, b in [(0.1, 0.001), (40, 33), (-97.4, -99)]:
        print log2_subtract(a, b), math.log(2 ** a - 2 ** b, 2.)
    from matplotlib import pyplot
    n = 20
    p = 0.4
    N = 2 ** 16
    g = pyplot.plot([2 ** log2_prob_binom(n, k, p) for k in xrange(0, n + 1)], '+r')
    count = collections.Counter(numpy.random.binomial(n, p, size=N))
    h = pyplot.plot([count[k] / float(N) for k in xrange(n + 1)], '.b')

    pyplot.show()

if __name__ == '__main__':
    main()