from dataclasses import dataclass, field
from typing import Sequence, List, Union
import numpy as np

from rfidam.inventory import get_rx_prob
from rfidam.protocol.symbols import InventoryFlag
from rfidam.protocol.protocol import Protocol
from rfidam.scenario import RoundSpec
from rfidam.statistics import StatsVector, group_round_values, \
    group_tag_values, count_averages


@dataclass
class ModelParams:
    protocol: Protocol
    arrival_interval: float
    time_in_area: float
    scenario: Sequence[RoundSpec]
    ber: float


@dataclass
class ScenarioInfo:
    # Number of rounds in the scenario (or extended scenario).
    # All attributes of `StatsVector` type should contain vectors `means`
    # and `errors` of shape `(scenario_length,)`.
    scenario_length: int

    # Estimated average round durations for rounds at offset `i` from
    # the start of scenario.
    round_durations: StatsVector

    # Number of active tags in rounds started with offset `i` from the
    # start of the scenario
    num_tags_active: StatsVector

    # Number of rounds the tag was active. Element at index `i` contains
    # average and std.dev. for number of rounds computed for tags those
    # arrived at round with offset `i` from the start of the scenario.
    num_rounds_active: StatsVector

    # Number of times the tag successfully transmitted its data.
    # Element at index `i` contains values for tags arrived at offset `i`
    # from the start of the round.
    num_times_identified: StatsVector

    # Identification probability for tags arrived at offset `i` from
    # the start of the scenario.
    id_probs: np.ndarray

    # Average identification probability over all tags.
    avg_id_prob: float


@dataclass
class Journal:
    #
    # Value per round:
    # ----------------
    # Round durations, incl. power-offs:
    round_durations: List[float] = field(default_factory=list)
    # Number of tags active in each round:
    num_tags_active: List[int] = field(default_factory=list)
    #
    # Value per tag:
    # --------------
    # Number of rounds tag took part in:
    num_rounds_active: List[int] = field(default_factory=list)
    # Round index the tag arrived in:
    first_round_index: List[int] = field(default_factory=list)
    # Number of times the tag was identified
    num_identified: List[int] = field(default_factory=list)
    #
    # Aggregated values:
    # ------------------
    # Number of tags identified
    n_identified: int = 0
    n_tags: int = 0

    @property
    def p_id(self) -> float:
        return float(self.n_identified) / self.n_tags if self.n_tags else 0.0


def build_scenario_info(
        journal: Union[Journal, Sequence[Journal]],
        sc_len: int) -> ScenarioInfo:
    """
    Build scenario info by averaging samples from one or several journals.
    """
    if not isinstance(journal, Journal):
        infos = [build_scenario_info(j, sc_len) for j in journal]

        def count_avg(stats_vectors: Sequence[StatsVector]) -> StatsVector:
            all_means = np.vstack([sv.means for sv in stats_vectors])
            all_errors = np.vstack([sv.errors for sv in stats_vectors])
            return StatsVector(all_means.mean(axis=0), all_errors.mean(axis=0))

        return ScenarioInfo(
            scenario_length=sc_len,
            round_durations=count_avg([i.round_durations for i in infos]),
            num_tags_active=count_avg([i.num_tags_active for i in infos]),
            num_rounds_active=count_avg([i.num_rounds_active for i in infos]),
            num_times_identified=count_avg(
                [i.num_times_identified for i in infos]),
            id_probs=(sum([i.id_probs for i in infos]) / len(infos)),
            avg_id_prob=np.asarray([i.avg_id_prob for i in infos]).mean(),
        )

    round_durations = group_round_values(journal.round_durations, sc_len)
    num_tags_active = group_round_values(journal.num_tags_active, sc_len)
    num_rounds_active = group_tag_values(
        journal.num_rounds_active, journal.first_round_index, sc_len)
    num_times_identified = group_tag_values(
        journal.num_identified, journal.first_round_index, sc_len)

    # To compute id_probs, we first need to build a vector of samples
    # with 0 - not identified, 1 - identified at least once.
    # Then we roll it using group_tag_values() and find means.
    was_identified_all = \
        (np.asarray(journal.num_identified) > 0).astype(float)
    was_identified = group_tag_values(
        was_identified_all, journal.first_round_index, sc_len)
    id_probs = np.asarray([np.mean(was_identified[i]) for i in range(sc_len)])
    avg_id_prob = np.mean(was_identified_all).item()

    return ScenarioInfo(
        scenario_length=sc_len,
        round_durations=count_averages(round_durations),
        num_tags_active=count_averages(num_tags_active),
        num_rounds_active=count_averages(num_rounds_active),
        num_times_identified=count_averages(num_times_identified),
        id_probs=id_probs,
        avg_id_prob=avg_id_prob)


@dataclass
class Tag:
    index: int
    time_in_area: float = 0.0
    flag: InventoryFlag = InventoryFlag.A
    n_id: int = 0


def simulate(params: ModelParams, *, only_id: bool = False,
             verbose: bool = False, n_tags: int = 2000) -> Journal:
    arrivals = params.arrival_interval * np.arange(1, n_tags+1)
    max_time_in_area = params.time_in_area
    sc_len: int = len(params.scenario)

    journal = Journal()

    # Define state:
    time = 0.0
    tags: List[Tag] = []         # tags in area
    round_index: int = 0         # current round index
    tag_index: int = 0           # index of the next tag to generate

    # MAX_ITER = 10
    num_iter = 0

    while tag_index < n_tags or tags:
        num_iter += 1
        if verbose:
            if num_iter > 0 and num_iter % 1000 == 0:
                print(f"* {num_iter} iterations passed, time = {time}, "
                      f"generated {tag_index}/{n_tags} tags: "
                      f"{[(tag.index, tag.time_in_area) for tag in tags]}")

        # Add new tags:
        while tag_index < n_tags and arrivals[tag_index] < time:
            created_at = arrivals[tag_index]
            tag = Tag(tag_index, time_in_area=(time - created_at))
            tags.append(tag)
            if not only_id:
                # Validate journal contains records up to this new tag:
                assert len(journal.num_rounds_active) == tag.index
                assert len(journal.first_round_index) == tag.index
                assert len(journal.num_identified) == tag.index
                # Add corresponding journal records:
                journal.num_rounds_active.append(0)
                journal.first_round_index.append(round_index)
                journal.num_identified.append(0)
            # Update offset and tag index:
            tag_index += 1
            journal.n_tags += 1

        # Remove too old tags:
        kept_tags = []
        for tag in tags:
            if tag.time_in_area < max_time_in_area:
                kept_tags.append(tag)
            elif tag.n_id > 0:
                journal.n_identified += 1
        tags = kept_tags

        round_duration = _sim_round(
            spec=params.scenario[round_index % sc_len],
            tags=tags,
            protocol=params.protocol,
            ber=params.ber,
            journal=journal,
            only_id=only_id)
        time += round_duration

        # Move tags:
        for tag in tags:
            tag.time_in_area += round_duration

        round_index += 1

    return journal


def _sim_round(spec: RoundSpec, tags: Sequence[Tag], protocol: Protocol,
               ber: float, journal: Journal, only_id: bool = False) -> float:
    """
    Simulate round and return its duration.

    Returns
    -------
    duration : float
        round duration
    """
    link = protocol.props
    rt_link, tr_link = protocol.rt_link, protocol.tr_link
    active_tags = [tag for tag in tags if tag.flag == spec.flag]

    # Extract values often used:
    t1, t2 = protocol.timings.t1, protocol.timings.t2

    # Record number of tags in area and number of active tags
    # to tag and round journals:
    if not only_id:
        for tag in active_tags:
            journal.num_rounds_active[tag.index] += 1

    # Select random slot for each tag:
    tags_slots = np.random.randint(0, link.n_slots, len(active_tags))

    # Compute number of tags in each slot:
    num_tags_per_slot = np.zeros(link.n_slots)
    for slot in tags_slots:
        num_tags_per_slot[slot] += 1

    n_empty = (num_tags_per_slot == 0).sum()
    n_collided = (num_tags_per_slot > 1).sum()

    # Compute round duration including all empty and collided slots,
    # Query and QueryRep commands. These durations don't depend on
    # success of particular replies transmissions.
    duration = (
            rt_link.query.duration +
            (link.n_slots - 1) * rt_link.query_rep.duration +
            n_empty * protocol.timings.t4 +
            n_collided * (t1 + t2 + tr_link.rn16.duration))

    # Now we model reply slots
    for tag, slot in zip(active_tags, tags_slots):
        # Skip, if there is a collision (duration is already computed above):
        if num_tags_per_slot[slot] > 1:
            continue

        # Attempt to transmit RN16:
        duration += t1 + tr_link.rn16.duration + t2
        if np.random.uniform() > get_rx_prob(tr_link.rn16, ber):
            # print("> failed to receive RN16")
            continue  # Error in RN16 reception

        # Reader transmits Ack, tag attempts to transmit EPCID.
        # Since tag transmitted EPCID, and we model no NACKs, invert tag flag.
        duration += rt_link.ack.duration + t1 + tr_link.epc.duration + t2
        tag.flag = tag.flag.invert()
        if np.random.uniform() > get_rx_prob(tr_link.epc, ber):
            # print("> failed to receive EPC")
            continue  # Error in EPCID reception

        if not link.use_tid:
            if not only_id:
                journal.num_identified[tag.index] += 1
            tag.n_id += 1
            continue  # Tag was identified, nothing more needed

        # Reader transmits Req_Rn, tag attempts to transmit Handle:
        duration += rt_link.req_rn.duration + t1 + tr_link.handle.duration + t2
        if np.random.uniform() > get_rx_prob(tr_link.handle, ber):
            # print("> failed to receive HANDLE")
            continue  # Error in Handle reception

        # Reader transmits Read, tag attempts to transmit Data:
        duration += rt_link.read.duration + t1 + tr_link.data.duration + t2
        if np.random.uniform() <= get_rx_prob(tr_link.data, ber):
            if not only_id:
                journal.num_identified[tag.index] += 1
            tag.n_id += 1
            continue
        # print("> failed to receive DATA")

    # If reader turns off after round, reset all tags flags and add
    # turn off period to round duration.
    if spec.turn_off:
        for tag in tags:
            tag.flag = InventoryFlag.A
        duration += link.t_off

    # Record round statistics:
    if not only_id:
        journal.num_tags_active.append(len(active_tags))
        journal.round_durations.append(duration)
    return duration
