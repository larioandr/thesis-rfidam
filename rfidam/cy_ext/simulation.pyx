#distutils: language = c++
from libcpp.list cimport list
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
# noinspection PyUnresolvedReferences
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np

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


cdef struct Descriptor:
    double t_off
    map[string, double] durations
    map[string, double] p_rx
    list[Tag*] tags
    bool use_tid
    int n_slots


# noinspection PyUnresolvedReferences
def simulate(params: ModelParams, verbose: bool = False) -> Journal:
    cdef int max_tags = len(params.arrivals)
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
    sim_desc.durations["t1"] = timings.t1
    sim_desc.durations["t2"] = timings.t2
    sim_desc.durations["t3"] = timings.t3
    sim_desc.durations["t4"] = timings.t4
    sim_desc.durations["query"] = rt_link.query.duration
    sim_desc.durations["query_rep"] = rt_link.query_rep.duration
    sim_desc.durations["ack"] = rt_link.ack.duration
    sim_desc.durations["rn16"] = tr_link.rn16.duration
    sim_desc.durations["epc"] = tr_link.epc.duration
    sim_desc.p_rx["rn16"] = get_rx_prob(tr_link.rn16, params.ber)
    sim_desc.p_rx["epc"] = get_rx_prob(tr_link.epc, params.ber)

    if props.use_tid:
        sim_desc.durations["req_rn"] = rt_link.req_rn.duration
        sim_desc.durations["read"] = rt_link.read.duration
        sim_desc.durations["handle"] = tr_link.handle.duration
        sim_desc.durations["data"] = tr_link.data.duration
        sim_desc.p_rx["handle"] = get_rx_prob(tr_link.handle, params.ber)
        sim_desc.p_rx["data"] = get_rx_prob(tr_link.data, params.ber)

    if verbose:
        print("DURATIONS:")
        print(sim_desc.durations)
        print("PROBABILITIES:")
        print(sim_desc.p_rx)

    journal = Journal()

    while tag_index < max_tags or not sim_desc.tags.empty():
        num_iter += 1
        if verbose:
            if num_iter > 0 and num_iter % 1000 == 0:
                print(f"* {num_iter} iterations passed, time = {time}, "
                      f"generated {tag_index}/{max_tags} tags")

        # Add new tags:
        while tag_index < max_tags and params.arrivals[tag_index] < time:
            created_at = params.arrivals[tag_index]
            tag = <Tag*> malloc(sizeof(Tag))
            tag.index = tag_index
            tag.time_in_area = time - created_at
            tag.flag = FLAG_A

            sim_desc.tags.push_back(tag)

            # Add corresponding journal records:
            journal.num_rounds_active.append(0)
            journal.first_round_index.append(round_index)
            journal.num_identified.append(0)

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
                if verbose:
                    print(f"tag {tag_ptr.index} departed at {time}")
                free(tag_ptr)
            else:
                inc(it)

        round_duration = _sim_round(
            sc_flags[round_index % sc_len],
            sc_turn_off[round_index % sc_len],
            &sim_desc, journal)
        time += round_duration

        # Move tags:
        it = sim_desc.tags.begin()
        while it != sim_desc.tags.end():
            tag_ptr = deref(it)
            tag_ptr.time_in_area += round_duration
            inc(it)

        round_index += 1

    return journal


cdef double _sim_round(int flag, bool turn_off, Descriptor* d, journal):
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
            journal.num_rounds_active[tag.index] += 1

    # Extract values often used:
    cdef double t1 = d.durations["t1"]
    cdef double t2 = d.durations["t2"]

    # Select random slot for each tag:
    cdef int n_active_tags = active_tags.size()
    cdef vector[int] tags_slots = np.random.randint(0, d.n_slots, n_active_tags)

    # Compute number of tags in each slot:
    cdef vector[int] n_tags_per_slot = np.zeros(d.n_slots)
    cdef int slot
    for slot in tags_slots:
        n_tags_per_slot[slot] += 1

    cdef int n_empty_slots = 0
    cdef int n_collided_slots = 0
    for slot in range(d.n_slots):
        if n_tags_per_slot[slot] == 0:
            n_empty_slots += 1
        elif n_tags_per_slot[slot] > 1:
            n_empty_slots += 1

    # Compute round duration including all empty and collided slots,
    # Query and QueryRep commands. These durations don't depend on
    # success of particular replies transmissions.
    cdef double duration = (
        d.durations["query"] +
        (d.n_slots - 1) * d.durations["query_rep"] +
        n_empty_slots * d.durations["t4"] +
        n_collided_slots * (t1 + t2 + d.durations["rn16"])
    )

    cdef double p

    # Now we model reply slots
    for i in range(n_active_tags):
        slot = tags_slots[i]
        tag = active_tags[i]
        if n_tags_per_slot[slot] > 1:
            continue

        # Attempt to transmit RN16:
        duration += t1 + t2 + d.durations["rn16"]
        p = np.random.uniform()
        if p > d.p_rx["rn16"]:
            continue  # Error in RN16 reception

        # Reader transmits Ack, tag attempts to transmit EPCID.
        # Since tag transmitted EPCID, and we model no NACKs, invert tag flag.
        duration += d.durations["ack"] + t1 + d.durations["epc"] + t2
        tag.flag = FLAG_B if tag.flag == FLAG_A else FLAG_A
        p = np.random.uniform()
        if p > d.p_rx["epc"]:
            continue  # Error in EPCID reception

        if not d.use_tid:
            journal.num_identified[tag.index] += 1
            continue  # Tag was identified, nothing more needed

        # Reader transmits Req_Rn, tag attempts to transmit Handle:
        duration += d.durations["req_rn"] + t1 + d.durations["handle"] + t2
        if np.random.uniform() > d.p_rx["handle"]:
            continue  # Error in Handle reception

        # Reader transmits Read, tag attempts to transmit Data:
        duration += d.durations["read"] + t1 + d.durations["data"] + t2
        if np.random.uniform() <= d.p_rx["data"]:
            journal.num_identified[tag.index] += 1
            continue

    # If reader turns off after round, reset all tags flags and add
    # turn off period to round duration.
    if turn_off:
        for tag in d.tags:
            tag.flag = FLAG_A
        duration += d.t_off

    # Record round statistics:
    journal.num_tags_active.append(n_active_tags)
    journal.round_durations.append(duration)
    return duration
