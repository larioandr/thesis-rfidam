from functools import cached_property, reduce
from typing import Tuple, Sequence, Optional

import numpy as np
from scipy import linalg
from rfidam.protocol.symbols import InventoryFlag

from rfidam.inventory import RoundModel, get_inventory_probs
from rfidam.protocol.protocol import Protocol
from rfidam.scenario import MarkedRoundSpec, RoundSpec, mark_scenario


class BgTransitions:
    def __init__(self, n_tags: int, n_tags_max: int,
                 inventory_probs: np.ndarray):
        """
        Factory for producing matrices for the background chain.

        Parameters
        ----------
        n_tags : int
            number of tags in the area when the step is performed
        n_tags_max : int
            maximum possible number of simultaneously existing tags
        inventory_probs: np.ndarray
            matrix where (n,m) element holds `P_n(m)` probability
            that m of n tags transmitted EPCID
        """
        self.n_tags = n_tags
        self.n_tags_max = n_tags_max
        self.inventory_probs = inventory_probs

    def _mat_draft(self):
        u = np.zeros((self.n_tags_max + 1, self.n_tags_max + 1))
        for i in range(self.n_tags + 1, self.n_tags_max + 1):
            u[i, i] = 1.0
        return u

    @cached_property
    def inventory_matrix(self):
        u = self._mat_draft()
        for i in range(self.n_tags + 1):
            for j in range(i + 1):
                u[i, j] = self.inventory_probs[i, i - j]
        return u

    @cached_property
    def target_switch_matrix(self):
        u = self._mat_draft()
        for i in range(self.n_tags + 1):
            u[i, self.n_tags - i] = 1.
        return u

    @cached_property
    def power_off_matrix(self):
        u = self._mat_draft()
        for i in range(self.n_tags + 1):
            u[i, self.n_tags] = 1.0
        return u

    @cached_property
    def arrival_matrix_a(self):
        u = self._mat_draft()
        for i in range(self.n_tags + 1):
            u[i, i + 1] = 1.0
        return u

    @cached_property
    def arrival_matrix_b(self):
        return np.identity(self.n_tags_max + 1)

    @cached_property
    def departure_matrix(self):
        u = self._mat_draft()
        for i in range(self.n_tags):
            u[i+1, i] = (i+1) / self.n_tags
            u[i, i] = 1 - i / self.n_tags
        return u


def estimate_rounds_props(
        scenario: Sequence[RoundSpec],
        protocol: Protocol,
        ber: float,
        arrival_interval: float,
        time_in_area: float,
        max_iters: int = 10,
        t0: Optional[float] = None
):
    models = [RoundModel(protocol, 0, ber), RoundModel(protocol, 1, ber)]
    t_off = protocol.props.t_off

    # Compute initial durations and number of active tags (assume that
    # initially in each round exactly one tag participates).
    durations = [
        models[1].round_duration + (t_off if spec.turn_off else 0)
        for spec in scenario
    ]
    n_active_tags = [np.array([0, 1]) for _ in scenario]

    t0 = t0 or 100 * arrival_interval

    n_iters = 1
    while n_iters <= max_iters:
        # 1) Mark scenario and build BG chain transitions:
        marked_scenario = mark_scenario(
            scenario,
            arrival_interval=arrival_interval,
            time_in_area=time_in_area,
            durations=durations,
            t0=t0)

        transitions = build_matrices(
            marked_scenario, BgTransitions,
            ber=ber,
            n_slots=protocol.props.n_slots,
            rn16_len=protocol.tr_link.rn16.bitlen)

        # if n_iters > 1:
        #     for i, trans in enumerate(transitions):
        #         print(f"D{i}\n", trans)
        #
        # 2) Compute number of active tags distributions:
        n_active_tags = get_num_active_tags_dists(transitions)

        # 3) Update rounds durations:

        # Each number of tags distribution has order N+1, so we can obtain
        # maximum number of tags from the shape:
        n_tags_max = n_active_tags[0].shape[0] - 1
        if len(models) <= n_tags_max:
            models.extend([
                RoundModel(protocol, n_tags, ber)
                for n_tags in range(len(models), n_tags_max + 1)
            ])
        ts = np.array([model.round_duration for model in models])
        updated_durations = [
            ts.dot(p) + (protocol.props.t_off if spec.turn_off else 0)
            for p, spec in zip(n_active_tags, marked_scenario)]

        # 4) Compute errors of durations estimations:
        errors = np.array([
            abs(d0 - d1) / d0 for d0, d1 in
            zip(durations, updated_durations)])
        durations = updated_durations
        if errors.mean() < .01:
            break

        n_iters += 1

    return {
        'durations': durations,
        'n_active_tags': n_active_tags,
        'n_iters': n_iters,
    }


def build_matrices(
        scenario: Sequence[MarkedRoundSpec],
        transitions_factory_class: type, *,
        ber: float,
        n_slots: int,
        rn16_len: int = 16
) -> Tuple[np.ndarray, ...]:
    """
    Build a sequence of matrices for the given marked scenario.

    Parameters
    ----------
    scenario
    transitions_factory_class
    ber
    n_slots
    rn16_len

    Returns
    -------
    matrices : tuple of ndarray
        number of matrices match the scenario length.
    """
    n_tags_set = {spec.num_tags for spec in scenario}
    n_tags_max = max(n_tags_set)
    inventory_probs = get_inventory_probs(
        n_tags_max, ber=ber, n_slots=n_slots, rn16_len=rn16_len)
    transitions = {
        n_tags: transitions_factory_class(n_tags, n_tags_max, inventory_probs)
        for n_tags in n_tags_set
    }
    matrices = []
    for i, spec in enumerate(scenario):
        next_spec = scenario[i+1] if i < len(scenario) - 1 else scenario[0]
        transition: BgTransitions = transitions[spec.num_tags]

        # Round always starts with inventory:
        mat = transition.inventory_matrix

        # Check whether tag will be added or removed:
        n_tags_delta = next_spec.num_tags - spec.num_tags
        if n_tags_delta == 1:
            tag_will_arrive, tag_will_depart = True, False
        elif n_tags_delta == 0:
            tag_will_arrive, tag_will_depart = False, False
        elif n_tags_delta == -1:
            tag_will_arrive, tag_will_depart = False, True
        else:
            raise RuntimeError(f"tag population changes by {n_tags_delta} "
                               f"tags, but at most 1 tag arrival or departure "
                               f"is allowed.")

        if spec.turn_off:
            # If reader turns off, then we are not interested in actual
            # flag inversion, since after power-on flag will be A.
            # However, if in next round target is B, then we invert after
            # power reset (that will make all tags inactive).
            mat = mat @ transition.power_off_matrix
            if next_spec.flag == InventoryFlag.B:
                mat = mat @ transition.target_switch_matrix
        else:
            # If no power off after round, then invert flag if needed.
            if next_spec.flag != spec.flag:
                mat = mat @ transition.target_switch_matrix

        if tag_will_arrive:
            if next_spec.flag == InventoryFlag.B:
                mat = mat @ transition.arrival_matrix_b
            elif next_spec.flag == InventoryFlag.A:
                mat = mat @ transition.arrival_matrix_a

        if tag_will_depart:
            mat = mat @ transition.departure_matrix

        # Add matrix to the result:
        matrices.append(mat)

    return tuple(matrices)


def get_num_active_tags_dists(
        chain: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """
    Compute stationary distribution of the background DTMCs.

    Parameters
    ----------
    chain : list-like of ndarray
        sequence of transition matrices for each round.

    Returns
    -------
    vectors : tuple of ndarray
        a tuple of stationary distributions for each round.
    """
    d0: np.ndarray = reduce(lambda x, y: x @ y, chain)

    # Solve system pi * D0 = pi, (pi, 1) = 1
    order = d0.shape[0]
    a = np.vstack((
        (d0.T - np.identity(order))[:-1],
        np.ones((1, order))
    ))
    b = np.array([0] * (order - 1) + [1])
    n_tags_dist = [linalg.solve(a, b)]
    for di in chain[:-1]:
        n_tags_dist.append(n_tags_dist[-1] @ di)
    # n_tags_dist[0] = n_tags_dist[-1] @ d0
    return tuple(n_tags_dist)
