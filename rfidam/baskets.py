"""
This module contains combinations routines for computing distributions of
balls in baskets.
"""
import numpy as np
from math import factorial
from functools import lru_cache as cache, cached_property
from math import comb
from collections import namedtuple
from typing import Sequence

from rfidam.baskets_mc import baskets_monte_carlo


def valez_alonso(m: int, n_balls: int, n_baskets: int):
    """
    Get probability that N tags transmit without collisions using formula
    from Valez-Alonso paper (https://www.mdpi.com/1424-8220/11/3/2946).

    WARNING: this formula doesn't work when n_tags_total > n_slots !!!

    Parameters
    ----------
    m : int
        number of balls to be put into separate urns
    n_balls : int
        total number of balls
    n_baskets : int
        number of urns

    Returns
    -------
    prob : float
    """
    if m > n_balls or m > n_baskets:
        return 0.0
    # try:
    k = factorial(n_baskets) / factorial(m) * (
        np.arange(1, n_balls+1) / n_baskets).prod()
    # k = (factorial(n_baskets) * factorial(n_balls) /
    #      (factorial(m) * np.power(n_baskets, n_balls)))
    # except ZeroDivisionError as er:
    #     print(f'Exception raised when n_balls={n_balls}, '
    #           f'n_baskets={n_baskets}')
    #     raise er
    sum_ = 0.0
    for i in range(n_balls - m + 1):
        sum_ += (pow(-1, i) *
                 pow(n_baskets - m - i, n_balls - m - i) / (
                     factorial(n_balls - m - i) *
                     factorial(i) *
                     factorial(n_baskets - m - i)))
    return k * sum_


@cache
def stirling2(n: int, k: int) -> int:
    """
    Compute Stirling number of the 2nd kind S(n, k).

    Find the number of ways to split n elements into k non-empty parts.

    Trivial cases:

    - S(n, 0) = 0 when n > 0
    - S(n, k) = 0 when k > n
    - S(n, n) = 1 when n >= 0, incl. S(0, 0) = 1
    - S(n, 1) = 1

    Parameters
    ----------
    n : int
        number of elements
    k : int
        number of partitions

    Returns
    -------
    s : int
    """
    if (k == 0 and n > 0) or k > n:
        return 0
    if k == n or k == 1:
        return 1
    sum_ = 0
    for i in range(0, k + 1):
        sum_ += pow(-1, i) * comb(k, i) * pow(k - i, n)
    return sum_ // np.math.factorial(k)


Occupancy = namedtuple('Occupancy', ['empty', 'single', 'many'])


def estimate_occupancy_rates(
        n_baskets: int, n_balls: int, n_iters: int = 20000) -> Occupancy:
    """
    Estimate occupancy of N baskets when K balls are put at random.

    More precise, it estimates probability mass functions of three random
    variables:

    - number of empty baskets
    - number of baskets with single ball
    - number of baskets with two or more balls

    While these variables are not independent, the routine returns only
    marginal distribution of each variable, since this is sufficient for the
    inventory round model.

    Each component of the returned `Occupancy` tuple contains a `numpy.ndarray`
    instance with i-th component equal to probability that i baskets has
    the corresponding type of occupancy (empty, single ball, many balls).

    This routine uses Monte-Carlo method: the process of balls distribution
    over baskets is repeated `n_iters` times, and occupancy is estimated
    from the values obtained.

    Parameters
    ----------
    n_baskets : int
    n_balls : int
    n_iters : int, optional (default = 20'000)

    Returns
    -------
    empty : numpy.ndarray
        probability mass function of the number of empty baskets
    single : numpy.ndarray
        probability mass function of the number of baskets with one ball
    many : numpy.ndarray
        probability mass function of the number of baskets with 2+ balls
    """
    empty, single, many = baskets_monte_carlo(n_baskets, n_balls, n_iters)
    return Occupancy(empty=empty, single=single, many=many)


def mean_num(pmf: Sequence[float]) -> float:
    """
    Helper routine for computing the average number by PMF.

    Returns `0*p[0] + 1*p[1] + 2*p[2] + ... + N*p[N]`.

    Parameters
    ----------
    pmf : sequence of float
        probabilities, `pmf[i]` is probability of value `i`.

    Returns
    -------
    mean : float
    """
    if not isinstance(pmf, np.ndarray):
        pmf = np.asarray(pmf)
    return pmf.dot(np.arange(len(pmf)))


def _get_empty_prob(m: int, n_baskets: int, n_balls: int) -> float:
    if n_baskets == 0:
        return 1.0 if m == 0 else 0.0
    p = stirling2(n_balls, n_baskets - m) / factorial(m)
    k = min(n_baskets, n_balls)
    p *= (np.arange(n_baskets, n_baskets - k, -1) / n_baskets).prod()
    p *= factorial(n_baskets - k)
    p /= pow(n_baskets, n_balls - k)
    return p


class BasketsOccupancyProblem:
    """
    Class representing a problem for finding probability distributions
    of the numbers of empty baskets, baskets with single ball and baskets
    with multiple balls.
    """
    N_ITER = 20000  # default number of iterations in Monte-Carlo method

    @staticmethod
    @cache
    def create(n_baskets: int, n_balls: int) -> 'BasketsOccupancyProblem':
        """
        Factory with cache to create the BasketOccupancyProblem instance.

        Using this method is preferred since it is cached, and so
        it won't take additional resources when estimating PMFs with
        Monte-Carlo method.
        """
        # print(f"called create with n_baskets={n_baskets}, n_balls={n_balls}")
        return BasketsOccupancyProblem(n_baskets, n_balls)

    def __init__(self, n_baskets: int, n_balls: int):
        self._n_baskets = n_baskets
        self._n_balls = n_balls

    @property
    def n_baskets(self):
        return self._n_baskets

    @property
    def n_balls(self):
        return self._n_balls

    @cached_property
    def _estimated_occupancy(self):
        return estimate_occupancy_rates(
            n_baskets=self.n_baskets,
            n_balls=self.n_balls,
            n_iters=BasketsOccupancyProblem.N_ITER)

    @cached_property
    def empty(self) -> np.ndarray:
        """
        Get probability distribution of the number of empty baskets.
        """
        # For large dimensions use Monte-Carlo method:
        if self.n_baskets > 8 or self.n_balls > 8:
            return self._estimated_occupancy.empty

        # For smaller values use analytic expression:
        probs = []
        n_baskets = self._n_baskets
        n_balls = self._n_balls
        for m in range(self.n_baskets + 1):
            probs.append(_get_empty_prob(m, n_baskets, n_balls))
        return np.asarray(probs)

    # @cached_property
    @property
    def single(self) -> np.ndarray:
        """
        Get PMF that exactly m balls will be put to separate urns.
        Other urns will be either empty, or contain more than one ball.
        """
        baskets = np.arange(self.n_baskets + 1)

        # In bad case, when formula from Valez-Alonso paper doesn't work,
        # use Monte-Carlo method to estimate probabilities:

        if self.n_baskets < self.n_balls or self.n_baskets > 8 or \
                self.n_balls > 8:
            return self._estimated_occupancy.single

        # FIXME: some error below, wrong result:
        # n_baskets, n_balls = self.n_baskets, self.n_balls
        # if self.n_baskets > 8 or self.n_balls > 8:
        #     return self._estimated_occupancy.single
        # probs = np.zeros(self.n_baskets + 1)
        # for i, m1 in enumerate(range(min(n_baskets, n_balls) + 1)):
        #     p = 0
        #     print("m1 =", m1)
        #     for m0 in range(self.n_baskets - m1 + 1):
        #         n_busy = n_baskets - m0
        #         print(f"m0={m0}: "
        #               f"p0={_get_empty_prob(m0, n_baskets, n_balls):.4f}, "
        #               f"p1={_get_empty_prob(m1, n_busy, n_balls-n_busy):.4f}")
        #         p += _get_empty_prob(m0, n_baskets, n_balls) * \
        #             _get_empty_prob(m1, n_baskets-m0, n_balls-(n_baskets-m0))
        #     probs[i] = p
        # return probs

        # Otherwise, use formula from Valez-Alonso paper:
        return np.asarray([
            valez_alonso(m, self.n_balls, self.n_baskets)
            # here we iterate through baskets, not balls: when m > n_balls,
            # zero will be returned, so no problem.
            for m in baskets
        ])

    @cached_property
    def avg_count(self) -> Occupancy:
        """
        Get average number of empty urns, urns with single ball and urns
        with multiple balls.
        """
        avg_empty = mean_num(self.empty)
        avg_single = mean_num(self.single)
        return Occupancy(
            empty=avg_empty,
            single=avg_single,
            many=(self.n_baskets - avg_empty - avg_single))
