import pytest
from numpy.testing import assert_allclose
from rfidam.protocol.symbols import InventoryFlag
from rfidam.scenario import get_arrivals_before, count_max_tags, \
    MarkedRoundSpec, parse_scenario, extend_scenario, mark_scenario


@pytest.mark.parametrize(
    'time, arrival_interval, time_in_area, arrivals, last_arrival', [
        # I. Inter-arrival time: 1.0, time in area: 2.3
        (0., 1., 2.3, (0.,), 0.),
        (.5, 1., 2.3, (0.,), 0.),
        (1., 1., 2.3, (0., 1.), 1.),
        (2.2, 1., 2.3, (0., 1., 2.,), 2.),
        (2.5, 1., 2.3, (1., 2.), 2.),
        (5.5, 1., 2.3, (4., 5.,), 5.,),
        # II. Inter-arrival time: 2.0, time in area: 1.4
        (0., 2., 1.4, (0.,), 0.),
        (1., 2., 1.4, (0.,), 0.),
        (1.5, 2., 1.4, (), 0.),
        (2.1, 2., 1.4, (2.,), 2.),
    ]
)
def test_get_arrivals_before(time, arrival_interval, time_in_area,
                             arrivals, last_arrival):
    real_arrivals, real_last_arrival = \
        get_arrivals_before(time, arrival_interval, time_in_area)
    assert_allclose(real_arrivals, arrivals)
    assert_allclose(real_last_arrival, last_arrival)


@pytest.mark.parametrize('arrival_interval, time_in_area, n', [
    (1., 2.3, 3),
    (2., 1.4, 1),
])
def test_count_max_tags(arrival_interval, time_in_area, n):
    assert count_max_tags(arrival_interval, time_in_area) == n


@pytest.mark.parametrize(
    'scenario,k,arrival_interval,time_in_area,durations,t0, marked_scenario', [
        # I. Inter-arrival time: 1.0, time in area: 2.3
        ('ABABx', 2, 1., 2.3, [.2, .2, .3, .4, .3, .2, .1, 0.42], 10., (
            MarkedRoundSpec(InventoryFlag.A, False, 3),  # t = 10.0
            MarkedRoundSpec(InventoryFlag.B, False, 3),  # t = 10.2
            MarkedRoundSpec(InventoryFlag.A, False, 2),  # t = 10.4
            MarkedRoundSpec(InventoryFlag.B, True, 2),   # t = 10.7
            MarkedRoundSpec(InventoryFlag.A, False, 3),  # t = 11.1
            MarkedRoundSpec(InventoryFlag.B, False, 2),  # t = 11.4
            MarkedRoundSpec(InventoryFlag.A, False, 2),  # t = 11.6
            MarkedRoundSpec(InventoryFlag.B, True, 2),   # t = 11.7
        )),
        # II. Inter-arrival time: 2.0, time in area: 1.4
        ('AAx', 3, 2., 1.4, [.5, .5, .4, .4, .6, .5], 21., (
            MarkedRoundSpec(InventoryFlag.A, False, 1),  # t = 21.0, la: 20.0
            MarkedRoundSpec(InventoryFlag.A, True, 0),   # t = 21.5, la: 20.0
            MarkedRoundSpec(InventoryFlag.A, False, 1),  # t = 22.0, la: 22.0
            MarkedRoundSpec(InventoryFlag.A, True, 1),   # t = 22.4, la: 22.0
            MarkedRoundSpec(InventoryFlag.A, False, 1),  # t = 22.8, la: 22.0
            MarkedRoundSpec(InventoryFlag.A, True, 0),   # t = 23.4, la: 22.0
        ))
    ]
)
def test_mark_scenario(scenario, k, arrival_interval, time_in_area, durations,
                       t0, marked_scenario):
    extended_scenario = extend_scenario(parse_scenario(scenario), k)
    real_marked_scenario = mark_scenario(
        extended_scenario, arrival_interval, time_in_area, durations, t0)
    assert real_marked_scenario == marked_scenario
