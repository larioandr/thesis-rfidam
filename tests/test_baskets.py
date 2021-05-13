import pytest
from numpy.testing import assert_allclose

from rfidam import baskets


@pytest.mark.parametrize('n_baskets, n_balls, m, prob', [
    # When no balls in the system, all bins are empty:
    (1, 0, 0, 1.0),
    (1, 0, 1, 0.0),
    # When one ball is in the system, it is always in one bin:
    (4, 1, 0, 0.0),
    (4, 1, 1, 1.0),
    # When there are two balls in the system, they are either collided, or not:
    (4, 2, 0, .25),
    (4, 2, 1, 0.0),
    (4, 2, 2, .75),
    (4, 2, 3, 0.0),
    # For eight values results were obtained from simulation:
    (8, 8, 0, 0.02084),
    (8, 8, 1, 0.09340),
    (8, 8, 2, 0.24024),
    (8, 8, 3, 0.22976),
    (8, 8, 4, 0.27988),
    (8, 8, 5, 0.06898),
    (8, 8, 6, 0.06452),
    (8, 8, 7, 0.0),
    (8, 8, 8, 0.00238),
    # For big dimensions:
    (16, 14, 0, 0.00062),
    (16, 14, 2, 0.02263),
    (16, 14, 7, 0.15525),
    (16, 14, 14, 0.00018),
])
def test_valez_alonso(n_baskets, n_balls, m, prob):
    """Validate Valez-Alonso formula.
    """
    actual = baskets.valez_alonso(m, n_baskets=n_baskets, n_balls=n_balls)
    assert_allclose(actual, prob, rtol=.05, atol=.001)


@pytest.mark.parametrize('n_baskets, n_balls, probs', [
    (8, 0, [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]),
    (8, 1, [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000]),
    (8, 4, [0.000, 0.000, 0.000, 0.000, 0.407, 0.495, 0.095, 0.002, 0.000]),
    (8, 8, [0.002, 0.067, 0.316, 0.420, 0.174, 0.020, 0.001, 0.000, 0.000]),
    (8, 10, [0.029, 0.218, 0.428, 0.269, 0.053, 0.003, 0.000, 0.000, 0.000]),
    # Many baskets:
    (16, 1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    (
        16, 14, [
            0, 0, 0, 0.004, 0.041, 0.160, 0.303, 0.295,
            0.152, 0.039, 0.005, 0, 0, 0, 0, 0, 0
        ]
    ), (
        16, 20, [
            0, 0.008, 0.054, 0.177, 0.296, 0.274, 0.142, 0.041,
            0.007, 0.001, 0, 0, 0, 0, 0, 0, 0
        ]
    ),
])
def test_baskets_occupancy_problem__empty(n_baskets, n_balls, probs):
    """Validate empty baskets PMF.
    """
    problem = baskets.BasketsOccupancyProblem(n_baskets, n_balls)
    assert_allclose(problem.empty, probs, rtol=.05, atol=.01)


@pytest.mark.parametrize('n_baskets, n_balls, probs', [
    (8, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0]),
    (8, 1, [0, 1, 0, 0, 0, 0, 0, 0, 0]),
    (8, 4, [0.043, 0.055, 0.492, 0, 0.41, 0, 0, 0, 0]),
    (8, 8, [0.021, 0.093, 0.239, 0.23, 0.28, 0.067, 0.067, 0, 0.002]),
    (8, 10, [0.023, 0.113, 0.234, 0.279, 0.215, 0.1, 0.032, 0.004, 0]),
    # Many baskets:
    (16, 1, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (
        16, 14, [
            0, 0.005, 0.023, 0.062, 0.128, 0.174, 0.215, 0.156,
            0.140, 0.049, 0.039, 0.004, 0.004, 0, 0, 0, 0
        ]
    ), (
        16, 20, [
            0.001, 0.006, 0.026, 0.072, 0.136, 0.191, 0.204, 0.169,
            0.109, 0.056, 0.022, 0.007, 0.001, 0, 0, 0, 0
        ]
    ),
])
def test_baskets_occupancy_problem__single(n_baskets, n_balls, probs):
    """Validate baskets with single ball PMF.
    """
    problem = baskets.BasketsOccupancyProblem(n_baskets, n_balls)
    assert_allclose(problem.single, probs, rtol=.05, atol=.01)


@pytest.mark.parametrize('n_baskets, n_balls, m0, m1, m2', [
    (8, 0, 8, 0, 0),
    (8, 1, 7, 1, 0),
    (8, 4, 4.68, 2.69, 0.63),
    (8, 8, 2.75, 3.14, 2.11),
    (8, 10, 2.1, 3., 2.89),
])
def test_baskets_occupancy_problem__means(n_baskets, n_balls, m0, m1, m2):
    """Validate mean numbers of empty bins, bins with single or many balls.
    """
    problem = baskets.BasketsOccupancyProblem(n_baskets, n_balls)
    count = problem.avg_count
    actual = count.empty, count.single, count.many
    assert_allclose(actual, (m0, m1, m2), rtol=.01)
