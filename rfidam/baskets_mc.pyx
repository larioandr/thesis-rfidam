import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


def baskets_monte_carlo(n_baskets, n_balls, n_iters):
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

    This routine uses Monte-Carlo method: the process of balls distribution
    over baskets is repeated `n_iters` times, and occupancy is estimated
    from the values obtained.

    Routine is implemented very low-level, all internal parts except
    getting random basket index are written in pure C-language.

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
    cdef int c_n_iters = n_iters
    cdef int c_n_baskets = n_baskets
    cdef int c_n_balls = n_balls

    cdef int* n_empty_array = <int*> malloc(c_n_iters * sizeof(int))
    cdef int* n_single_array = <int*> malloc(c_n_iters * sizeof(int))
    cdef int* n_many_array = <int*> malloc(c_n_iters * sizeof(int))
    cdef int* baskets = <int*> malloc(c_n_baskets * sizeof(int))

    cdef int i, j, k, basket_index, num_balls

    cdef np.ndarray random_indexes = np.random.randint(
        0,
        c_n_baskets,
        size=(c_n_balls * c_n_iters))

    k = 0  # index of the random number

    # Simulate balls distribution:
    for i in range(c_n_iters):
        for j in range(c_n_baskets):
            baskets[j] = 0

        for j in range(c_n_balls):
            basket_index = random_indexes[k]
            baskets[basket_index] += 1
            k += 1

        # Count number of baskets empty, with a single or many balls
        n_empty_array[i] = 0
        n_single_array[i] = 0
        n_many_array[i] = 0

        for j in range(c_n_baskets):
            num_balls = baskets[j]
            if num_balls == 0:
                n_empty_array[i] += 1
            elif num_balls == 1:
                n_single_array[i] += 1
            else:
                n_many_array[i] += 1

    # Compute numbers of occasions that `m` baskets were empty, occupied with
    # single or many balls, for each `m = 0 ... n_baskets`:
    cdef np.ndarray empty_counts = np.zeros((n_baskets + 1,), dtype=int)
    cdef np.ndarray single_counts = np.zeros((n_baskets + 1,), dtype=int)
    cdef np.ndarray many_counts = np.zeros((n_baskets + 1,), dtype=int)

    for i in range(c_n_iters):
        empty_counts[n_empty_array[i]] += 1
        single_counts[n_single_array[i]] += 1
        many_counts[n_many_array[i]] += 1

    free(n_empty_array)
    free(n_single_array)
    free(n_many_array)
    free(baskets)

    # Compute and return probabilities mass functions:
    return (empty_counts / n_iters,
            single_counts / n_iters,
            many_counts / n_iters)
