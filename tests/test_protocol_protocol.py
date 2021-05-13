import pytest
from numpy.testing import assert_allclose
from rfidam.protocol.responses import TagPreamble, TagFrame, Rn16, Epc,\
    Handle, Data
from rfidam.protocol.symbols import TagEncoding, DR, min_t1, max_t1, min_t2, \
    max_t2, t3
from rfidam.protocol.commands import ReaderPreamble, ReaderSync, \
    ReaderFrame, Query, QueryRep, Ack, ReqRn, Read

from rfidam.protocol import protocol


TARI = 6.25e-6
RTCAL = 20e-6
TRCAL = 30e-6
M = TagEncoding.M2
DR_ = DR.DR_643
TREXT = False
Q = 5
T_OFF = .2
USE_TID = True
N_DATA_WORDS = 4
N_EPCID_BYTES = 12


#
# TEST LinkProps
# ---------------------------------------------------------------------------
@pytest.fixture
def link_props():
    return protocol.LinkProps(
        tari=TARI, rtcal=RTCAL, trcal=TRCAL, m=M, dr=DR_, trext=TREXT, q=Q,
        t_off=T_OFF, use_tid=USE_TID, n_data_words=N_DATA_WORDS,
        n_epcid_bytes=N_EPCID_BYTES)


def test_link_props__n_slots(link_props):
    """Test LinkProps n_slots.property.
    """
    assert link_props.n_slots == 2**Q


#
# TEST RTLink
# ---------------------------------------------------------------------------
def test_rtlink__props(link_props):
    rtlink = protocol.RTLink(link_props)
    assert rtlink.props is link_props


@pytest.fixture
def rtlink(link_props):
    return protocol.RTLink(link_props)


def test_rtlink__preamble(rtlink):
    assert isinstance(rtlink.preamble, ReaderPreamble)
    assert rtlink.preamble is rtlink.preamble, "preamble not cached"
    assert rtlink.preamble.tari == TARI
    assert rtlink.preamble.rtcal == RTCAL
    assert rtlink.preamble.trcal == TRCAL


def test_rtlink__sync(rtlink):
    assert isinstance(rtlink.sync, ReaderSync)
    assert rtlink.sync is rtlink.sync, "SYNC not cached"
    assert rtlink.sync.tari == TARI
    assert rtlink.sync.rtcal == RTCAL


def test_rtlink__query(rtlink):
    assert isinstance(rtlink.query, ReaderFrame)
    assert rtlink.query is rtlink.query, "Query frame not cached"
    assert rtlink.query.preamble is rtlink.preamble, "preamble not cached"
    assert isinstance(rtlink.query.msg, Query)
    assert rtlink.query.msg.q == Q
    assert rtlink.query.msg.m == M
    assert rtlink.query.msg.dr == DR_
    assert rtlink.query.msg.trext == TREXT


def test_rtlink__query_rep(rtlink):
    assert isinstance(rtlink.query_rep, ReaderFrame)
    assert rtlink.query_rep is rtlink.query_rep, "QueryRep frame not cached"
    assert rtlink.query_rep.preamble is rtlink.sync, "preamble not cached"
    assert isinstance(rtlink.query_rep.msg, QueryRep)


def test_rtlink__ack(rtlink):
    assert isinstance(rtlink.ack, ReaderFrame)
    assert rtlink.ack is rtlink.ack, "Ack frame not cached"
    assert rtlink.ack.preamble is rtlink.sync, "preamble not cached"
    assert isinstance(rtlink.ack.msg, Ack)


def test_rtlink__req_rn(rtlink):
    assert isinstance(rtlink.req_rn, ReaderFrame)
    assert rtlink.req_rn is rtlink.req_rn, "ReqRn frame not cached"
    assert rtlink.req_rn.preamble is rtlink.sync, "preamble not cached"
    assert isinstance(rtlink.req_rn.msg, ReqRn)


def test_rtlink__read(rtlink):
    assert isinstance(rtlink.read, ReaderFrame)
    assert rtlink.read is rtlink.read, "Read frame not cached"
    assert rtlink.read.preamble is rtlink.sync, "preamble not cached"
    assert isinstance(rtlink.read.msg, Read)
    assert rtlink.read.msg.wordcnt == N_DATA_WORDS


#
# TEST TRLink
# ---------------------------------------------------------------------------
def test_trlink__props(link_props):
    trlink = protocol.TRLink(link_props)
    assert trlink.props is link_props
    assert_allclose(trlink.blf, DR_.ratio / TRCAL)


@pytest.fixture
def trlink(link_props):
    return protocol.TRLink(link_props)


def test_trlink__preamble(trlink):
    assert isinstance(trlink.preamble, TagPreamble)
    assert trlink.preamble is trlink.preamble, "tag preamble not cached"
    assert trlink.preamble.trext == TREXT
    assert trlink.preamble.m == M
    assert_allclose(trlink.preamble.blf, trlink.blf)


def test_trlink__rn16(trlink):
    assert isinstance(trlink.rn16, TagFrame)
    assert trlink.rn16 is trlink.rn16, "Rn16 frame not cached"
    assert trlink.rn16.preamble is trlink.preamble, "preamble not cached"
    assert isinstance(trlink.rn16.msg, Rn16)


def test_trlink__epc(trlink):
    assert isinstance(trlink.epc, TagFrame)
    assert trlink.epc is trlink.epc, "Epc frame not cached"
    assert trlink.epc.preamble is trlink.preamble, "preamble not cached"
    assert isinstance(trlink.epc.msg, Epc)
    assert trlink.epc.msg.bitlen == N_EPCID_BYTES * 8 + 32


def test_trlink__handle(trlink):
    assert isinstance(trlink.handle, TagFrame)
    assert trlink.handle is trlink.handle, "Handle frame not cached"
    assert trlink.handle.preamble is trlink.preamble, "preamble not cached"
    assert isinstance(trlink.handle.msg, Handle)


def test_trlink__data(trlink):
    assert isinstance(trlink.data, TagFrame)
    assert trlink.data is trlink.data, "Data frame not cached"
    assert trlink.data.preamble is trlink.preamble, "preamble not cached"
    assert isinstance(trlink.data.msg, Data)
    assert trlink.data.msg.bitlen == N_DATA_WORDS * 16 + 33


#
# TEST Timings
# ---------------------------------------------------------------------------
def test_timings(link_props):
    timings = protocol.Timings(link_props)
    # Validate basic properties:
    assert timings.props is link_props
    blf = DR_.ratio / TRCAL
    assert_allclose(timings.blf, blf)
    # Validate T1, T2, T3, T4:
    assert min_t1(RTCAL, blf) <= timings.t1 <= max_t1(RTCAL, blf)
    assert min_t2(blf) <= timings.t2 <= max_t2(blf)
    assert_allclose(timings.t3, t3())
    assert timings.t4 >= timings.t1 + timings.t2


#
# TEST Protocol
# ---------------------------------------------------------------------------
def test_protocol(link_props):
    proto_inst = protocol.Protocol(link_props)
    assert proto_inst.props is link_props
    # Validate RTLink:
    assert isinstance(proto_inst.rt_link, protocol.RTLink)
    assert proto_inst.rt_link.props is link_props
    # Validate TRLink:
    assert isinstance(proto_inst.tr_link, protocol.TRLink)
    assert proto_inst.tr_link.props is link_props
    # Validate Timings:
    assert isinstance(proto_inst.timings, protocol.Timings)
    assert proto_inst.timings.props is link_props
