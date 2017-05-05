# Notations:
# p = prior P(y = 1), that is y ~ bernoulli(p)
# l_list[i] = lambda_i
# n_list[i] = n_i = amount of bits in x randomized as bernoulli(0.5 * (1 - lambda_i)) ^ y
# n = sum(n_list)
# alpha_list[i] = alpha_i = amount of 1 bits out of the above defined n_i.
import itertools
import collections
import math
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from utils import singletonbyparameters
from utils import utils


class Problem_k_D(object):
    def __init__(self, n_list, l_list, p):
        assert 0 < p < 1
        for l_i in l_list:
            assert 0 < l_i < 1
        for n_i in n_list:
            assert n_i > 0
        assert len(l_list) == len(n_list)
        self.n_list = list(n_list)
        self.l_list = list(l_list)
        self.p = p

    @property
    def n(self):
        return sum(self.n_list)

    def k(self):
        return len(self.n_list)


class Point(object):
    __metaclass__ = singletonbyparameters.SingletonByParameters

    def __init__(self, problem, alpha_list):
        assert isinstance(problem, Problem_k_D)
        assert len(alpha_list) == len(problem.n_list)
        self.alpha_list = tuple(alpha_list)
        for alpha_i, n_i in zip(self.alpha_list, problem.n_list):
            assert 0 <= alpha_i <= n_i
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
        p = self.problem.p
        prob0 = math.log(1 - p, 2)
        prob1 = math.log(p, 2)
        for n_i, l_i, alpha_i in zip(self.problem.n_list, self.problem.l_list, self.alpha_list):
            prob0 += utils.log2_prob_binom(n_i, alpha_i,
                                           0.5 * (1 - l_i))
            prob1 += utils.log2_prob_binom(n_i, alpha_i,
                                           0.5 * (1 + l_i))
        return prob0, prob1

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
        for i in xrange(len(self.alpha_list)):
            for d in (-1, 1):
                alpha_i = self.alpha_list[i] + d
                if 0 <= alpha_i <= self.problem.n_list[i]:
                    new_list = list(self.alpha_list)
                    new_list[i] += d
                    res.append(Point(self.problem, tuple(new_list)))
        return res

    def __getitem__(self, i):
        return self.alpha_list[i]


def central_points(problem, modolus):
    assert isinstance(problem, Problem_k_D)
    for alpha_tuple in itertools.product(*[range(n_i + 1) for n_i in problem.n_list]):
        if sum(alpha_tuple) % 2 == modolus:
            x = Point(problem, alpha_tuple)
            if x.neighbors_sign() is None:
                yield x


def create_weighted_graph(problem):
    central = list(central_points(problem, 1))
    graph = nx.DiGraph()
    for c in central:
        for n in c.neighbors():
            if n.is_positive():
                graph.add_edge(n, c)
                graph.node[n]["type"] = "R"
                assert "color" not in graph.node[n] or graph.node[n]["color"] == "green"
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
    assert isinstance(problem, Problem_k_D)
    assert problem.k() == 2
    fig, ax = plt.subplots()
    ax.set_xlim((-1, problem.n_list[0] + 1))
    ax.set_ylim((-1, problem.n_list[1] + 1))
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 11,
            }
    g = weighted_inference_graph
    for n in g.nodes():
        circle1 = plt.Circle((n[0], n[1]), 0.05, color=g.node[n]["color"])
        ax.add_artist(circle1)
        ax.text(n[0], n[1], g.node[n].get("label", ""), fontdict=font)
    for a, b in g.edges():
        if a[0] == b[0]:
            ax.arrow(a[0], a[1] + 0.2, 0, b[1] - a[1] - 0.4,
                 head_width=0.1, head_length=0.1, fc='k', ec='k')
        else:
            assert a[1] == b[1]
            if a[0] < b[0]:
                ax.arrow(a[0] + 0.2, a[1], b[0] - a[0] - 0.4, 0,
                     head_width=0.1, head_length=0.1, fc='k', )
            else:
                ax.arrow(a[0] - 0.2, a[1], b[0] - a[0] + 0.4, 0,
                     head_width=0.1, head_length=0.1, fc='k', )


def plot_3d_graph(problem, weighted_inference_graph):
    assert isinstance(problem, Problem_k_D)
    assert problem.k() == 3
    g = weighted_inference_graph

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points_by_color = collections.defaultdict(list)
    for n in g.nodes():
        points_by_color[g.node[n]["color"]].append(n)
    for color, points in points_by_color.iteritems():
        ax.scatter([p[0] for p in points],
                   [p[1] for p in points],
                   [p[2] for p in points], c=color, marker='o')
    x, y, z, w, u, v = [], [], [], [],[], []
    for a, b in g.edges():
        for i in xrange(3):
            if a[i] != b[i]:
                break
        assert a[i] != b[i]
        source = [a[j] for j in xrange(3)]
        target = [b[j] for j in xrange(3)]
        if a[i] > b[i]:
            source[i] -= 0.2
        else:
            source[i] += 0.2
        x.append(source[0])
        y.append(source[1])
        z.append(source[2])
        u.append(target[0] - source[0])
        v.append(target[1] - source[1])
        w.append(target[2] - source[2])
    ax.quiver(x, y, z, u, v, w, pivot="tail", length=0.7, arrow_length_ratio=0.05)


def create_weighted_corruption_graph(weighted_inference_graph):
    g = weighted_inference_graph
    graph = nx.Graph()
    for n, attrs in g.node.iteritems():
        if attrs["type"] in ("R", "L"):
            t, w = attrs["type"], attrs["weight"]
            graph.add_node(n, type=t, w=abs(w))
    for n in graph.nodes():
        if graph.node[n]["type"] == "R":
            for x in g.edge[n]:
                for m in g.edge[x]:
                    graph.add_edge(n, m)
    return graph


def is_node_dismissable1(graph, n):
    common_neighbors = [m for m in graph.neighbors(n) if len(graph.neighbors(m)) > 1]
    distance_2_neighbors = [m for m in graph.nodes() if set(graph.neighbors(n)) & set(graph.neighbors(m))]
    if len(graph.neighbors(n)) > 0 and (len(common_neighbors) <= 1 or len(distance_2_neighbors) == 1):
        return True
    else:
        # It might be dismissable for other reasons - but we neglect them for the moment.
        return False


def soaking_neighbor(graph, n):
    if len(graph.neighbors(n)) == 0:
        return None
    neighbor2neighbors = {m: graph.neighbors(m) for m in graph.neighbors(n)}
    soaking, neighbors_of_soaking = sorted(neighbor2neighbors.items(), key=lambda (m, l): len(l))[0]
    for l in neighbor2neighbors.itervalues():
        if not set(l).issuperset(neighbors_of_soaking):
            return None
    return soaking


def is_node_dismissable2(graph, n):
    if soaking_neighbor(graph, n) is not None:
        return True
    return False


def nodes2dismiss(weighted_corruption_graph):
    g = weighted_corruption_graph
    for n in g.nodes():
        if is_node_dismissable1(g, n):
            yield n


def use_path(graph, n, m):
    if graph.node[m]["w"] <= graph.node[n]["w"]:
        argmin = m
        argmax = n
    else:
        argmin = n
        argmax = m
    graph.node[argmax]["w"] -= graph.node[argmin]["w"]
    graph.remove_node(argmin)
    if graph.node[argmax]["w"] == 0:
        graph.remove_node(argmax)


def dismiss_node(graph, n):
    common_neighbors = [m for m in graph.neighbors(n) if len(graph.neighbors(m)) > 1]
    unique_neighbors = [m for m in graph.neighbors(n) if m not in common_neighbors]
    for m in unique_neighbors:
        use_path(graph, n, m)
        if n not in graph.node:
            return
    for m in common_neighbors:
        use_path(graph, n, m)
        if n not in graph.node:
            return


def dismiss_node2(graph, n):
    soaking = soaking_neighbor(graph, n)
    assert soaking is not None
    use_path(graph, n, soaking)


def solve_graph(weighted_corruption_graph):
    g = weighted_corruption_graph
    while 1:
        to_dismiss = list(nodes2dismiss(g))
        if len(to_dismiss) == 0:
            any = False
            for n in g.nodes():
                if n in g:
                    if is_node_dismissable2(g, n):
                        dismiss_node2(g, n)
                        any = True
            if not any:
                break
        for n in to_dismiss:
            if n in g:
                dismiss_node(g, n)
    return g


def is_graph_solveable(unweighted_corruption_graph):
    pass


def main2d():
    n_list = [10, 10]
    p = 0.5
    for i, lambdas in enumerate([(0.2, 0.1), (0.1, 0.1), (0.01, 0.02), (0.9, 0.3)]):
        problem = Problem_k_D(n_list, lambdas, p)
        graph = create_weighted_graph(problem)
        corruption = create_weighted_corruption_graph(graph)
        leftovers = solve_graph(corruption)
        for n in leftovers.nodes():
            graph.node[n]["label"] = "%.2f" % (math.log(leftovers.node[n]["w"]))
        plot_2d_graph(problem, graph)
        plt.title("n_list=%s, lambda=%s, p=%.2f" % (n_list, lambdas, p))
        plt.savefig("../output/sample_2D_{:02d}.png".format(i))
        plt.close()


def main3d():
    n_list, lambdas, p = [5, 5, 5], [0.1, 0.2, 0.02], 0.6
    problem = Problem_k_D(n_list, lambdas, p)
    graph = create_weighted_graph(problem)
    plot_3d_graph(problem, graph)
    plt.title("n_list=%s, lambda=%s, p=%.2f" % (n_list, lambdas, p))
    plt.show()


if __name__ == '__main__':
    main2d()
    main3d()