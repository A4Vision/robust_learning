# Notations:
# p = prior P(y = 1), that is y ~ bernoulli(p)
# l1 = lambda1, l2 = lambda2
# n1 = amount of bits in x randomized as bernoulli(0.5 * (1 - lambda1)) ^ y
# n2 = amount of bits in x randomized as bernoulli(0.5 * (1 - lambda2)) ^ y
# n = n1 + n2
# alpha = amount of 1 bits out of the above defined n1.
# beta  = amount of 1 bits out of the above defined n2.
import networkx as nx
import math
from utils import singletonbyparameters
from utils import utils
import itertools
from matplotlib import pyplot as plt


class Problem2D(object):
    def __init__(self, n1, n2, l1, l2, p):
        assert 0 < l1 < 1
        assert 0 < l2 < 1
        assert 0 < p < 1
        assert n1 > 0
        assert n2 > 0
        self.n1 = n1
        self.n2 = n2
        self.l1 = l1
        self.l2 = l2
        self.p = p

    @property
    def n(self):
        return self.n1 + self.n2


class Point2D(object):
    __metaclass__ = singletonbyparameters.SingletonByParameters

    def __init__(self, problem, alpha, beta):
        assert isinstance(problem, Problem2D)
        assert 0 <= alpha <= problem.n1
        assert 0 <= beta <= problem.n2
        self.alpha = alpha
        self.beta = beta
        self.problem = problem
        log2_prob_0, log2_prob_1 = self._calc_log2_probs()
        if log2_prob_0 > log2_prob_1:
            self._sign = 1
            # log2(P(y=0, point) - P(y=1, point))
            self._log2_da = utils.log2_subtract(log2_prob_0, log2_prob_1)
        elif log2_prob_0 < log2_prob_1:
            self._sign = -1
            self._log2_da = utils.log2_subtract(log2_prob_1, log2_prob_0)
        else:
            self._sign = 0
            self._log2_da = 0

    def _calc_log2_probs(self):
        """
        :return: log2(P(y=0, point)), log2(P(y=1, point))
        """
        n1, n2, p = self.problem.n1, self.problem.n2, self.problem.p
        log2_binom1 = utils.log2_prob_binom(n1, self.alpha,
                                            0.5 * (1 - self.problem.l1))
        log2_binom2 = utils.log2_prob_binom(n2, self.beta,
                                            0.5 * (1 - self.problem.l2))
        part1 = log2_binom1 + log2_binom2 + math.log(1 - p, 2)
        log2_binom1 = utils.log2_prob_binom(n1, self.alpha,
                                            0.5 * (1 + self.problem.l1))
        log2_binom2 = utils.log2_prob_binom(n2, self.beta,
                                            0.5 * (1 + self.problem.l2))
        part2 = log2_binom1 + log2_binom2 + math.log(p, 2)
        return part1, part2

    def log2_da(self):
        assert self._sign != 0
        return self._log2_da

    def da(self):
        return self._sign * 2 ** self._log2_da

    def is_positive(self):
        return self._sign > 0

    def is_zero(self):
        return self._sign == 0

    def sign(self):
        return self._sign

    def is_negative(self):
        return self._sign < 0

    def neighbors_sign(self):
        signs = {n.sign() for n in self.neighbors()}
        if -1 not in signs:
            return 1
        elif 1 not in signs:
            return -1
        else:
            return None

    def neighbors(self):
        res = []
        for d1, d2 in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            alpha = self.alpha + d1
            beta = self.beta + d2
            if 0 <= alpha <= self.problem.n1 and 0 <= beta <= self.problem.n2:
                res.append(Point2D(self.problem, alpha, beta))
        return res


def central_points(problem_2d, modolus):
    assert isinstance(problem_2d, Problem2D)
    for i, j in itertools.product(range(problem_2d.n1 +1), range(problem_2d.n2 + 1)):
        if (i + j) % 2 == modolus:
            x = Point2D(problem_2d, i, j)
            if x.neighbors_sign() is None:
                yield x


def create_2d_weighted_graph(problem):
    """
    :param p: Aprior P(y=0)
    :param n: Amount of random bits, len(x)
    :param l1: lambda1
    :param l2: lambda2
    :return:
    """
    central = list(central_points(problem, 1))
    graph = nx.DiGraph()
    for c in central:
        for n in c.neighbors():
            if n.is_positive():
                graph.add_edge(n, c)
                graph.node[n]["type"] = "R"
                graph.node[n]["color"] = "green"
            elif n.is_negative():
                graph.add_edge(c, n)
                graph.node[n]["type"] = "L"
                graph.node[n]["color"] = "red"
            else:
                continue
            graph.node[n]["weight"] = n.da()
        graph.node[c]["color"] = "blue"
        graph.node[c]["type"] = "M"
    return graph


def plot_2d_graph(problem, weighted_inference_graph):
    fig, ax = plt.subplots()
    ax.set_xlim((-1, problem.n1 + 1))
    ax.set_ylim((-1, problem.n2 + 1))
    g = weighted_inference_graph
    for n in g.nodes():
        circle1 = plt.Circle((n.alpha, n.beta), 0.05, color=g.node[n]["color"])
        ax.add_artist(circle1)
    for a, b in g.edges():
        if a.alpha == b.alpha:
            ax.arrow(a.alpha, a.beta + 0.2, 0, b.beta - a.beta - 0.4,
                 head_width=0.1, head_length=0.1, fc='k', ec='k')
        else:
            assert a.beta == b.beta
            if a.alpha < b.alpha:
                ax.arrow(a.alpha + 0.2, a.beta, b.alpha - a.alpha - 0.4, 0,
                     head_width=0.1, head_length=0.1, fc='k', )
            else:
                ax.arrow(a.alpha - 0.2, a.beta, b.alpha - a.alpha + 0.4, 0,
                     head_width=0.1, head_length=0.1, fc='k', )

    plt.show()


def solve_graph(weighted_corruption_graph):
    pass


def is_graph_solveable(unweighted_corruption_graph):
    pass


def main():
    problem = Problem2D(20, 20, 0.1, 0.2, 0.6)
    graph = create_2d_weighted_graph(problem)
    plot_2d_graph(problem, graph)


if __name__ == '__main__':
    main()