"""
Module contains classes for simplification of protocol description.

Protocol is represented with four classes:

- `LinkProps`: contains symbols durations, coding and round settings, etc.;
- `RTLink`: reader-tag link, provides reader frames;
- `TRLink`: tag-reader link, provides tag frames;
- `Timings`: provides various timeouts;
- `Protocol`: joins all these things above.

Since some parameters are not very important, they are joined in `defaults`
dictionary.
"""
from dataclasses import dataclass
from functools import cached_property

from rfidam.protocol.symbols import TagEncoding, DR, Session, InventoryFlag, \
    Sel, Bank, get_blf, nominal_t1, max_t2, min_t2, t3, t4
from rfidam.protocol.commands import ReaderPreamble, ReaderSync, \
    ReaderFrame, Query, QueryRep, Ack, ReqRn, Read
from rfidam.protocol.responses import TagPreamble, TagFrame, Rn16, Epc, \
    Handle, Data


# Dictionary with default protocol settings of less-significant parameters.
# While these parameters may be used by the algorithm, their exact values
# in commands and replies is not very important. They are assumed
# to be hardcoded to these values in commands and responses.
defaults = {
    'session': Session.S1,
    'sel': Sel.ALL,
    'target': InventoryFlag.A,
    'crc5': 127,
    'crc16': 0xAAAA,
    'rn': 0x5555,
    'wordptr': 0,
    'bank': Bank.TID,
    'pc': 0,
}


@dataclass
class LinkProps:
    """
    Link properties with most significant parameters.

    These parameters define the commands and responses durations, so they
    are used in preambles and syncs. EPCID and TID sizes are also here.
    """
    tari: float
    rtcal: float
    trcal: float
    m: TagEncoding
    dr: DR
    trext: bool
    q: int
    use_tid: bool
    t_off: float = .1  # duration of reader power off intervals
    n_data_words: int = 4
    n_epcid_bytes: int = 12

    @cached_property
    def n_slots(self) -> int:
        """Number of slots in the inventory round.
        """
        return 2**self.q


class RTLink:
    """
    Reader-tag link, provides reader frames.

    All command frames returned in RTLink instance properties are cached,
    so once created they are always the same (until user invalidates cache)
    in the same run.

    To make things more easy, some fields are shared among commands, for
    example - CRC-5 and CRC-16. All frames except frame with Query share
    exactly the same SYNC preamble instance.
    """
    def __init__(self, props: LinkProps):
        self.props = props

    @cached_property
    def preamble(self) -> ReaderPreamble:
        """Get Query preamble.
        """
        return ReaderPreamble(
            tari=self.props.tari,
            rtcal=self.props.rtcal,
            trcal=self.props.trcal)

    @cached_property
    def sync(self) -> ReaderSync:
        """Get other than Query commands preamble (SYNC).
        """
        return ReaderSync(tari=self.props.tari, rtcal=self.props.rtcal)

    @cached_property
    def query(self) -> ReaderFrame:
        """Get reader frame with Query command.
        """
        return ReaderFrame(self.preamble, Query(
            q=self.props.q,
            m=self.props.m,
            dr=self.props.dr,
            trext=self.props.trext,
            sel=defaults['sel'],
            session=defaults['session'],
            target=defaults['target'],
            crc5=defaults['crc5']))

    @cached_property
    def query_rep(self) -> ReaderFrame:
        """Get reader frame with QueryRep command.
        """
        return ReaderFrame(self.sync, QueryRep(session=defaults['session']))

    @cached_property
    def ack(self) -> ReaderFrame:
        """Get reader frame with Ack command.
        """
        return ReaderFrame(self.sync, Ack(rn=defaults['rn']))

    @cached_property
    def req_rn(self) -> ReaderFrame:
        """Get reader frame with Req_Rn command.
        """
        return ReaderFrame(self.sync, ReqRn(
            rn=defaults['rn'],
            crc16=defaults['crc16']))

    @cached_property
    def read(self) -> ReaderFrame:
        """Get reader frame with Read command.
        """
        return ReaderFrame(self.sync, Read(
            bank=defaults['bank'],
            wordptr=defaults['wordptr'],
            wordcnt=self.props.n_data_words,
            rn=defaults['rn'],
            crc16=defaults['crc16']
        ))


class TRLink:
    """
    Class provides tag-reader link frames, preambles and symbols.

    In the same manner as in RTLink, all properties are cached, and
    all tag frames share exactly the same preamble. Most significant
    properties come from `LinkProps` instance, while less important
    are taken from `defaults` dictionary.
    """
    def __init__(self, props: LinkProps):
        self._props = props

    @cached_property
    def blf(self) -> float:
        """Backscatter link frequency.
        """
        return get_blf(self._props.dr, self._props.trcal)

    @property
    def props(self) -> LinkProps:
        """Link properties.
        """
        return self._props

    @cached_property
    def preamble(self) -> TagPreamble:
        """Tag frame preamble.
        """
        return TagPreamble(
            m=self.props.m,
            trext=self.props.trext,
            blf=self.blf)

    @cached_property
    def rn16(self) -> TagFrame:
        """Tag frame with RN16 response.
        """
        return TagFrame(self.preamble, Rn16(rn=defaults['rn']))

    @cached_property
    def epc(self) -> TagFrame:
        """Tag frame with response to Ack command (EPCID + PC + CRC).
        """
        epcid = 'A0' * self._props.n_epcid_bytes
        return TagFrame(self.preamble, Epc(
            epcid=epcid,
            pc=defaults['pc'],
            crc16=defaults['crc16']))

    @cached_property
    def handle(self) -> TagFrame:
        """Tag frame with response to Req_Rn command (RN16 + CRC).
        """
        return TagFrame(self.preamble, Handle(
            rn=defaults['rn'],
            crc16=defaults['crc16']))

    @cached_property
    def data(self) -> TagFrame:
        """Tag frame with response to Read command carrying data words.
        """
        words = 'CDEF' * self._props.n_data_words
        return TagFrame(self.preamble, Data(
            words=words,
            rn=defaults['rn'],
            crc16=defaults['crc16'],
            header=0))


class Timings:
    """
    Provides various timeouts (T1, T2, T3, T4).

    Standard specified limits for timeouts T1 and T2, here we use their
    average values. All timeouts are computed once and cached, so access
    to them is reasonably fast.
    """
    def __init__(self, props: LinkProps):
        self._props = props

    @property
    def props(self) -> LinkProps:
        """Link properties.
        """
        return self._props

    @cached_property
    def blf(self) -> float:
        """Backscatter link frequency.
        """
        return get_blf(self._props.dr, self._props.trcal)

    @cached_property
    def t1(self) -> float:
        """Time after the end of reader command and start of tag response.
        """
        return nominal_t1(self.props.rtcal, self.blf)

    @cached_property
    def t2(self) -> float:
        """Time after the end of tag response and start of next command.
        """
        return 0.5 * (min_t2(self.blf) + max_t2(self.blf))

    @cached_property
    def t3(self) -> float:
        """Additional time reader waits after T1 before next command.
        """
        return t3()

    @cached_property
    def t4(self) -> float:
        """Minimum time between reader commands.
        """
        return max(t4(self.props.rtcal), self.t1 + self.t3)


class Protocol:
    """
    Unites RTLink, TRLink and Timings in a single object representing
    EPC Class 1 Generation 2 protocol.
    """
    def __init__(self, props: LinkProps):
        self._props = props

    @cached_property
    def props(self) -> LinkProps:
        """Link properties.
        """
        return self._props

    @cached_property
    def rt_link(self) -> RTLink:
        """Reader-tag link object with reader frames.
        """
        return RTLink(self._props)

    @cached_property
    def tr_link(self):
        """Tag-reader link object with tag response frames.
        """
        return TRLink(self._props)

    @cached_property
    def timings(self):
        """Timings object with protocol timeouts.
        """
        return Timings(self._props)
