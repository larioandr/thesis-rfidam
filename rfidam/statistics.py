from typing import Any, List, Sequence
from collections import namedtuple
import numpy as np


def group_round_values(values: List[Any], sc_len: int) -> List[List[Any]]:
    """
    Group values measured in each round by their offset in scenario.

    Parameters
    ----------
    values : list
        values measured in each sequential round, i.e. i-th value was measured
        in round #i.
    sc_len : int
        length of the scenario

    Returns
    -------
    list of lists
        i-th item contains a list of values collected in rounds with offset
        equal to i from the start of the scenario.
    """
    grouped_values = [list() for _ in range(sc_len)]
    for i, value in enumerate(values):
        grouped_values[i % sc_len].append(value)
    return grouped_values


def group_tag_values(values: List[Any], first_round_index: List[int],
                     sc_len: int) -> List[List[Any]]:
    """
    Group values measured for each tag by the offset of the round the tag
    arrived in.

    Parameters
    ----------
    values : list
        values measured for each tag, i-th value is measured for i-th tag
    first_round_index : list of int
        i-th value is the round index the i-th tag arrived in
    sc_len : int
        length of the scenario

    Returns
    -------
    list of lists
        i-th item contains values collected for tags arrived in the i-th
        round from the start of the scenario.
    """
    grouped_values = [list() for _ in range(sc_len)]
    for value, round_index in zip(values, first_round_index):
        grouped_values[round_index % sc_len].append(value)
    return grouped_values


StatsVector = namedtuple('StatsVector', ('means', 'errors'))


def count_averages(all_samples: Sequence[Sequence[float]]) -> StatsVector:
    """
    Take samples grouped by round offset and compute mean and std.devs.

    Accepts a sequence of sequences of samples, grouped by offset from the
    scenario start. For example, let's say original samples sequence `A`
    contained 13 samples, and scenario length is 3. Then `all_samples` will
    look like this:

    +--------+--------------------------------+     +----+----+
    | Offset | Samples                        |     | Ms | Es |
    +--------+--------------------------------+     +----+----+
    | 0      | A[0], A[3], A[6], A[9], A[12]  |     | M0 | E0 |
    | 1      | A[1], A[4], A[7], A[10], A[13] | =>  | M1 | E1 |
    | 2      | A[2], A[5], A[8], A[11]        |     | M2 | E2 |
    +--------+--------------------------------+     +----+----+

    This function computes mean and standard error of each sub-sequence. For
    the example above, it will return arrays `means` and `errors` of size 3,
    where `means[0]` is equal to `M0 = mean(A[0], A[3], A[6], A[9], A[12])`.

    Parameters
    ----------
    all_samples : sequence of sequence of floats
        samples grouped by the offset from the start of the scenario

    Returns
    -------
    stats : StatsVector
        means and errors computed from samples for each offset.
        If `all_samples` was of length `N`, then `stats.means` and
        `stats.errors` will be also of length `N`.
    """
    all_samples = [np.asarray(samples) if len(samples) > 0 else np.zeros(1)
                   for samples in all_samples]
    means = np.asarray([samples.mean() for samples in all_samples])
    errors = np.asarray([samples.std() for samples in all_samples])
    return StatsVector(means, errors)
