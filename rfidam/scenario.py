from collections import deque
from dataclasses import dataclass
from typing import Tuple, Iterable

from rfidam.protocol.symbols import InventoryFlag


@dataclass
class RoundSpec:
    flag: InventoryFlag
    turn_off: bool = False

    def as_tuple(self):
        return self.flag, self.turn_off


@dataclass
class MarkedRoundSpec(RoundSpec):
    n_tags: int = 0
    n_arrived: int = 0
    n_departed: int = 0

    def as_tuple(self):
        return super().as_tuple() + (self.n_tags,)


def parse_scenario(s: str) -> Tuple[RoundSpec, ...]:
    """
    Parse string that encodes a scenario into a sequence of RoundSpec.

    Scenario specification looks like "AABBxABx", where symbols 'A' and 'B'
    specify inventory flag, and 'x' indicates that reader turns off in after
    the round. Denoting round spec as (X, e), where X = A,B and e = True
    iff reader turns off.

    Example
    -------
    >>> parse_scenario("AB")
    ((A, False), (B, False))

    >>> parse_scenario("ABBxAAx")
    ((A, False), (B, False), (B, True), (A, False), (A, True))

    Parameters
    ----------
    s : str

    Returns
    -------
    scenario : tuple of RoundSpec
    """
    if s == "":
        return ()
    pos, max_pos = 0, len(s) - 1
    scenario = []
    while pos <= max_pos:
        flag = InventoryFlag.parse(s[pos])
        if pos < max_pos and s[pos + 1] == 'x':
            turn_off = True
            pos += 2
        else:
            turn_off = False
            pos += 1
        scenario.append(RoundSpec(flag, turn_off))
    return tuple(scenario)


def extend_scenario(scenario: Iterable[RoundSpec],
                    n: int) -> Tuple[RoundSpec, ...]:
    """
    Repeat scenario n times.

    Parameters
    ----------
    scenario : sequence of RoundSpec
    n : int

    Returns
    -------
    extended_scenario : tuple of RoundSpec
    """
    scenario = tuple(scenario)
    return scenario * n


def get_arrivals_before(time: float, arrival_interval: float,
                        time_in_area: float) -> Tuple[Tuple[float, ...], float]:
    """
    Get a tuple of times, when the tags those are in the area at the given
    moment of time (`time)`, arrived.


    Parameters
    ----------
    time : float
    arrival_interval : float
    time_in_area : float

    Returns
    -------
    arrivals: tuple of float
    last_arrival: float
    """
    num_arrived = int(time // arrival_interval)
    last_arrival = num_arrived * arrival_interval
    arrival = last_arrival
    arrivals_list = []
    while arrival >= 0 and arrival + time_in_area > time:
        arrivals_list.append(arrival)
        departure = arrival + time_in_area
        arrival -= arrival_interval
    arrivals_list.reverse()
    return tuple(arrivals_list), last_arrival


def count_max_tags(arrival_interval: float, time_in_area: float) -> int:
    """
    Get maximum number of tags in area.

    Parameters
    ----------
    arrival_interval : float
    time_in_area : float

    Returns
    -------
    n : int
    """
    return int(time_in_area // arrival_interval) + 1


def mark_scenario(
        scenario: Iterable[RoundSpec],
        arrival_interval: float,
        time_in_area: float,
        durations: Iterable[float],
        t0: float) -> Tuple[MarkedRoundSpec]:
    """
    Build a marked scenario - sequence of specs with number of tags in area.

    Parameters
    ----------
    scenario : sequence of round specs
    arrival_interval : float
        interval between successive tags arrivals
    time_in_area : float
        time the tag spends in the area
    durations : sequence of float
        rounds durations
    t0 : float
        time when scenario starts

    Returns
    -------
    sequence of marked round specifications
    """
    arrivals, last_arrival = \
        get_arrivals_before(t0, arrival_interval, time_in_area)
    time = t0
    departure_queue = deque([arrival + time_in_area for arrival in arrivals])
    next_arrival = last_arrival + arrival_interval

    marked_scenario = []
    n_tags = len(departure_queue)

    for spec, duration in zip(scenario, durations):
        spec = MarkedRoundSpec(
            flag=spec.flag,
            turn_off=spec.turn_off,
            n_tags=n_tags)
        time += duration

        # Check whether arrival appeared during the previous round:
        if time >= next_arrival:
            departure_queue.append(next_arrival + time_in_area)
            spec.n_arrived = 1
            next_arrival += arrival_interval
            if time > next_arrival:
                raise ValueError(
                    f"too short inter-arrival interval {arrival_interval}: "
                    f"two or more arrivals in one round not supported")
        else:
            spec.n_arrived = 0

        # Check whether departure appeared during the previous round:
        if len(departure_queue) > 0 and time >= departure_queue[0]:
            departure_queue.popleft()
            spec.n_departed = 1
            if len(departure_queue) > 0 and time > departure_queue[0]:
                raise ValueError(
                    f"too short inter-arrival interval {arrival_interval}: "
                    f"two or more departures in one round not supported")
        else:
            spec.n_departed = 0

        marked_scenario.append(spec)
        n_tags += (spec.n_arrived - spec.n_departed)

    return tuple(marked_scenario)
