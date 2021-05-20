from libcpp.list cimport list
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool
from libc.stdlib cimport malloc, free
# noinspection PyUnresolvedReferences
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
cimport numpy as np

from rfidam.simulation import ModelParams, Journal
from rfidam.protocol.symbols import InventoryFlag
from rfidam.inventory import get_rx_prob


cdef int FLAG_A = 0
cdef int FLAG_B = 1

cdef enum Symbols:
    T1, T2, T3, T4,
    QUERY, QUERY_REP, ACK, REQ_RN, READ,
    RN16, EPC, HANDLE, DATA

cdef struct Tag:
    int index
    double time_in_area
    int flag
    int n_id


cdef struct Descriptor:
    double t_off
    map[Symbols, double] durations
    map[Symbols, double] p_rx
    list[Tag*] tags
    bool use_tid
    int n_slots


cdef struct CJournal:
    vector[double] rounds_durations
    vector[int] num_tags_active
    vector[int] num_rounds_active
    vector[int] first_round_index
    vector[int] num_identified
    int n_identified
    int n_tags


cdef struct Rnd:
    vector[double] values
    int next_index


cdef init_rnd(Rnd* rnd, int sz):
    rnd.values = np.random.uniform(0, 1, sz)
    rnd.next_index = 0


cdef double next_rnd(Rnd* rnd):
    cdef double ret
    if rnd.next_index == <int>rnd.values.size():
        init_rnd(rnd, <int>rnd.values.size())
    ret = rnd.values[rnd.next_index]
    rnd.next_index += 1
    return ret


# noinspection PyUnresolvedReferences
def simulate(params: ModelParams, only_id: bool = False,
             verbose: bool = False, n_tags: int = 1000) -> Journal:
    cdef double max_time_in_area = params.time_in_area
    cdef int sc_len = len(params.scenario)

    cdef double time = 0.0      # current simulation time
    cdef int round_index = 0    # current round index
    cdef int tag_index = 0      # next tag index
    cdef int num_iter = 0

    # Auxiliary variables
    cdef double created_at
    cdef double round_duration
    cdef list[Tag*].iterator it
    cdef Tag* tag_ptr

    # Define random numbers generator for fast access:
    cdef Rnd rnd
    init_rnd(&rnd, 1000)

    # Define arrival timestamps:
    cdef vector[float] arrivals = \
        params.arrival_interval * np.arange(1, n_tags+1)

    # To prevent calling Python, extract flags and turn-offs from scenario:
    cdef vector[int] sc_flags
    cdef vector[bool] sc_turn_off

    for spec in params.scenario:
        sc_flags.push_back(FLAG_A if spec.flag == InventoryFlag.A else FLAG_B)
        sc_turn_off.push_back(spec.turn_off)

    # Define descriptor that will be passed during simulation:
    props, timings, rt_link, tr_link = \
        params.protocol.props, params.protocol.timings, \
        params.protocol.rt_link, params.protocol.tr_link

    cdef Descriptor sim_desc
    sim_desc.t_off = props.t_off
    sim_desc.use_tid = props.use_tid
    sim_desc.n_slots = props.n_slots
    sim_desc.durations[T1] = timings.t1
    sim_desc.durations[T2] = timings.t2
    sim_desc.durations[T3] = timings.t3
    sim_desc.durations[T4] = timings.t4
    sim_desc.durations[QUERY] = rt_link.query.duration
    sim_desc.durations[QUERY_REP] = rt_link.query_rep.duration
    sim_desc.durations[ACK] = rt_link.ack.duration
    sim_desc.durations[RN16] = tr_link.rn16.duration
    sim_desc.durations[EPC] = tr_link.epc.duration
    sim_desc.p_rx[RN16] = get_rx_prob(tr_link.rn16, params.ber)
    sim_desc.p_rx[EPC] = get_rx_prob(tr_link.epc, params.ber)

    # noinspection DuplicatedCode
    if props.use_tid:
        sim_desc.durations[REQ_RN] = rt_link.req_rn.duration
        sim_desc.durations[READ] = rt_link.read.duration
        sim_desc.durations[HANDLE] = tr_link.handle.duration
        sim_desc.durations[DATA] = tr_link.data.duration
        sim_desc.p_rx[HANDLE] = get_rx_prob(tr_link.handle, params.ber)
        sim_desc.p_rx[DATA] = get_rx_prob(tr_link.data, params.ber)

    if verbose:
        print("DURATIONS:")
        print(sim_desc.durations)
        print("PROBABILITIES:")
        print(sim_desc.p_rx)

    cdef CJournal journal
    journal.n_identified = 0
    journal.n_tags = 0

    cdef bool c_only_id = only_id

    while tag_index < n_tags or not sim_desc.tags.empty():
        num_iter += 1
        if verbose:
            if num_iter > 0 and num_iter % 1000 == 0:
                print(f"* {num_iter} iterations passed, time = {time}, "
                      f"generated {tag_index}/{n_tags} tags")

        # Add new tags:
        while tag_index < n_tags and arrivals[tag_index] < time:
            created_at = arrivals[tag_index]
            tag = <Tag*> malloc(sizeof(Tag))
            tag.index = tag_index
            tag.time_in_area = time - created_at
            tag.flag = FLAG_A
            tag.n_id = 0
            journal.n_tags += 1

            sim_desc.tags.push_back(tag)

            # Add corresponding journal records:
            if not c_only_id:
                journal.num_rounds_active.push_back(0)
                journal.first_round_index.push_back(round_index)
                journal.num_identified.push_back(0)

            # Update offset and tag index:
            if verbose:
                print(f"tag {tag_index} arrived at {time}")
            tag_index += 1

        # print(sim_desc.tags)

        # Remove too old tags:
        it = sim_desc.tags.begin()
        while it != sim_desc.tags.end():
            tag_ptr = deref(it)
            if tag_ptr.time_in_area > max_time_in_area:
                it = sim_desc.tags.erase(it)
                if tag_ptr.n_id > 0:
                    journal.n_identified += 1
                if verbose:
                    print(f"tag {tag_ptr.index} departed at {time}")
                free(tag_ptr)
            else:
                inc(it)

        round_duration = _sim_round(
            sc_flags[round_index % sc_len],
            sc_turn_off[round_index % sc_len],
            &sim_desc, &journal, &rnd, c_only_id)
        time += round_duration

        # Move tags:
        it = sim_desc.tags.begin()
        while it != sim_desc.tags.end():
            tag_ptr = deref(it)
            tag_ptr.time_in_area += round_duration
            inc(it)

        round_index += 1

    return convert_journal(&journal)


cdef double _sim_round(int flag, bool turn_off, Descriptor* d,
                       CJournal* journal, Rnd* rnd, bool only_id):
    """
    Simulate round and return its duration.

    Returns
    -------
    duration : float
        round duration
    """
    cdef vector[Tag*] active_tags
    cdef Tag* tag
    cdef int i

    for tag in d.tags:
        if tag.flag == flag:
            active_tags.push_back(tag)
            # Record number of tags in area and number of active tags
            # to tag and round journals:
            if not only_id:
                journal.num_rounds_active[tag.index] += 1

    # Extract values often used:
    cdef double t1 = d.durations[T1]
    cdef double t2 = d.durations[T2]

    # Select random slot for each tag:
    cdef int n_active_tags = active_tags.size()
    cdef vector[int] tags_slots
    cdef vector[int] n_tags_per_slot
    cdef int slot
    cdef int n_empty_slots = 0
    cdef int n_collided_slots = 0

    if n_active_tags > 0:
        tags_slots = np.random.randint(0, d.n_slots, n_active_tags)
        n_tags_per_slot = vector[int](d.n_slots, 0)

        for slot in tags_slots:
            n_tags_per_slot[slot] += 1

        for slot in range(d.n_slots):
            if n_tags_per_slot[slot] == 0:
                n_empty_slots += 1
            elif n_tags_per_slot[slot] > 1:
                n_collided_slots += 1
    else:
        n_empty_slots = d.n_slots

    # Compute round duration including all empty and collided slots,
    # Query and QueryRep commands. These durations don't depend on
    # success of particular replies transmissions.
    cdef double duration = (
        d.durations[QUERY] +
        (d.n_slots - 1) * d.durations[QUERY_REP] +
        n_empty_slots * d.durations[T4] +
        n_collided_slots * (t1 + t2 + d.durations[RN16])
    )

    # Now we model reply slots
    for i in range(n_active_tags):
        slot = tags_slots[i]
        tag = active_tags[i]
        if n_tags_per_slot[slot] > 1:
            continue

        # Attempt to transmit RN16:
        duration += t1 + t2 + d.durations[RN16]
        if next_rnd(rnd) > d.p_rx[RN16]:
            continue  # Error in RN16 reception

        # Reader transmits Ack, tag attempts to transmit EPCID.
        # Since tag transmitted EPCID, and we model no NACKs, invert tag flag.
        duration += d.durations[ACK] + t1 + d.durations[EPC] + t2
        tag.flag = FLAG_B if tag.flag == FLAG_A else FLAG_A
        if next_rnd(rnd) > d.p_rx[EPC]:
            continue  # Error in EPCID reception

        if not d.use_tid:
            tag.n_id += 1
            if not only_id:
                journal.num_identified[tag.index] += 1
            continue  # Tag was identified, nothing more needed

        # Reader transmits Req_Rn, tag attempts to transmit Handle:
        duration += d.durations[REQ_RN] + t1 + d.durations[HANDLE] + t2
        if next_rnd(rnd) > d.p_rx[HANDLE]:
            continue  # Error in Handle reception

        # Reader transmits Read, tag attempts to transmit Data:
        duration += d.durations[READ] + t1 + d.durations[DATA] + t2
        if next_rnd(rnd) <= d.p_rx[DATA]:
            tag.n_id += 1
            if not only_id:
                journal.num_identified[tag.index] += 1
            continue

    # If reader turns off after round, reset all tags flags and add
    # turn off period to round duration.
    if turn_off:
        for tag in d.tags:
            tag.flag = FLAG_A
        duration += d.t_off

    # Record round statistics:
    if not only_id:
        journal.num_tags_active.push_back(n_active_tags)
        journal.rounds_durations.push_back(duration)
    return duration


cdef convert_journal(CJournal* cj):
    pj = Journal()
    pj.round_durations = [x for x in cj.rounds_durations]
    pj.num_tags_active = [int(x) for x in cj.num_tags_active]
    pj.num_rounds_active = [int(x) for x in cj.num_rounds_active]
    pj.first_round_index = [int(x) for x in cj.first_round_index]
    pj.num_identified = [int(x) for x in cj.num_identified]
    pj.n_identified = cj.n_identified
    pj.n_tags = cj.n_tags
    return pj
