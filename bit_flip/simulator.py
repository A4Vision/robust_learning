import random
import numpy


def simulate(p, lambdas, attacker, defender):
    """
    Clarifies the problem definition.
    :param p:
    :param lambdas:
    :param attacker:
    :param defender:
    :return:
    """
    assert 0 < p < 1
    assert isinstance(lambdas, numpy.ndarray)
    n = len(lambdas)
    y = int(random.random() < p)
    assert numpy.logical_and(0 < lambdas, lambdas < 1).all()
    probs = 0.5 * (1 - lambdas)
    x = numpy.array(numpy.random.random(size=n) < probs).astype(numpy.int) ^ y
    i = attacker(x)
    assert 0 <= i < n
    x[i] ^= 1
    z = defender(x)
    return y, x, i, z


print simulate(0.3, numpy.array([0.1, 0.99, 0.02]), lambda x: 0, lambda x: int(numpy.sum(x) > len(x) / 2.))
