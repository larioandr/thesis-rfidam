from typing import Tuple
from functools import cached_property, lru_cache as cache

from rfidam.protocol.symbols import DR, TagEncoding, Bank, InventoryFlag, \
    Sel, Session, encode, CommandCode


class ReaderSync:
    """
    Reader SYNC preamble.

    This short preamble consists of `Delimiter`, `Tari` and `RTcal`.
    SYNC is used before every command except Query.

    All parameters should be given in seconds.

    Parameters
    ----------
    tari : float
        data-0 symbol duration. According to standard, may take values
        6.25e-6, 12.5e-6, 18.75e-6 or 25e-6.
    rtcal : float
        Reader-Tag calibration symbol with duration data-0 plus data-1.
        According to standard, between `2.5 x Tari ... 3.0 x Tari`.
    delim : float, optional
        Delimiter, default: 12.5e-6.
    """
    def __init__(self, tari: float, rtcal: float, delim: float = 12.5e-6):
        self._tari = tari
        self._rtcal = rtcal
        self._delim = delim

    @property
    def delim(self) -> float:
        return self._delim

    @property
    def rtcal(self) -> float:
        return self._rtcal

    @property
    def tari(self) -> float:
        return self._tari

    @property
    def data0(self) -> float:
        return self._tari

    @cached_property
    def data1(self) -> float:
        return self.rtcal - self.tari

    @cached_property
    def duration(self) -> float:
        return self.delim + self.tari + self.rtcal

    def __repr__(self) -> str:
        return f"(Sync: delim={self.delim * 1e6:g}us " \
            f"tari={self.tari*1e6:g}us rtcal={self.rtcal*1e6:g}us)"


class ReaderPreamble(ReaderSync):
    """
    Reader full preamble that is used in frames with Query command.

    In contrast to SYNC, preamble contains TRcal symbol that is used
    by tags to compute their BLF.

    All parameters values should be given in seconds.

    Parameters
    ----------
    tari : float
        data-0 symbol duration. According to standard, may take values
        6.25e-6, 12.5e-6, 18.75e-6 or 25e-6.
    rtcal : float
        Reader-Tag calibration symbol with duration data-0 plus data-1.
        According to standard, between `2.5 x Tari ... 3.0 x Tari`.
    trcal : float
        Tag-Reader calibration symbol. According to standard, its value
        is between `1.1 x RTcal ... 3 x RTcal`.
    delim : float, optional
        Delimiter, default: 12.5e-6.
    """
    def __init__(self, tari: float, rtcal: float, trcal: float,
                 delim: float = 12.5e-6):
        self._trcal = trcal
        super().__init__(tari, rtcal, delim)

    @property
    def trcal(self):
        return self._trcal

    @cached_property
    def duration(self) -> float:
        return self.delim + self.tari + self.rtcal + self.trcal

    def __repr__(self) -> str:
        return f"(Preamble: delim={self.delim*1e6:g}us " \
               f"tari={self.tari*1e6:g}us rtcal={self.rtcal*1e6:g}us " \
               f"trcal={self.trcal*1e6:g}us)"


class Command:
    """
    Abstract base class for reader commands.

    Any derived class should implement two methods:

    - `_encode_body()`: returns a string of `01` that represents bits of
        the encoded command body.
    - `_str_body()`: returns a human-readable string representing the command.

    Commands themself do not provide duration property, since preamble
    is needed to know data-0 and data-1 durations.
    """
    def __init__(self, code: CommandCode):
        """
        Command constructor.

        Parameters
        ----------
        code : CommandCode
        """
        self._code = code

    @property
    def code(self) -> CommandCode:
        raise self._code

    @cached_property
    def name(self) -> str:
        return CommandCode.get_name_for(self._code)

    @cached_property
    def encoded(self) -> str:
        """
        Get bit string representing the encoded command.
        """
        return CommandCode.encode(self._code) + self._encode_body()

    @cached_property
    def bitlen(self) -> int:
        """
        Get the number of bits in the encoded command.
        """
        return len(self.encoded)

    @cache
    def count_bits(self) -> Tuple[int]:
        """
        Get a tuple with the number of 0s and 1s in the encoded command.
        """
        return tuple(len([x for x in self.encoded if x == b]) for b in '01')

    def _encode_body(self) -> str:
        """
        Get encoded command representation. Must be implemented in children.
        """
        raise NotImplementedError()

    def _str_body(self) -> str:
        """
        Get human-readable representation. Must be implemented in children.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f'({self.name}: {self._str_body()})'


class Query(Command):
    """
    Query command.
    """
    def __init__(self, q: int, m: TagEncoding, dr: DR = DR.DR_8,
                 trext: bool = False, sel: Sel = Sel.ALL,
                 session: Session = Session.S0,
                 target: InventoryFlag = InventoryFlag.A, crc5: int = 0x00):
        """
        Constructor.

        Parameters
        ----------
        q : int
        m : TagEncoding
            tag encoding, FM0, M2, M4 or M8.
        dr : DR, optional
            division ratio, 8 or 64/3. Default: DR.DR_8
        trext : bool, optional
            flag indicating whether the tag need to send extended preamble.
            Default: False
        session : Session, optional
            session to use in the inventory round, default: S0
        target : InventoryFlag, optional
            what session inventory flag the tag should have to take part
            in the inventory round. Default: A
        crc5 : int, optional
            checksum CRC-5, default: 0x00
        """
        self._dr = dr
        self._m = m
        self._trext = trext
        self._sel = sel
        self._session = session
        self._target = target
        self._q = q
        self._crc5 = crc5
        super().__init__(CommandCode.QUERY)

    @property
    def q(self) -> int:
        return self._q

    @property
    def m(self) -> TagEncoding:
        return self._m

    @property
    def dr(self) -> DR:
        return self._dr

    @property
    def trext(self) -> bool:
        return self._trext

    @property
    def sel(self) -> Sel:
        return self._sel

    @property
    def session(self) -> Session:
        return self._session

    @property
    def target(self) -> InventoryFlag:
        return self._target

    @property
    def crc5(self) -> int:
        return self._crc5

    def _encode_body(self) -> str:
        return encode(self.dr) + encode(self.m) + encode(self.trext) + \
               encode(self.sel) + encode(self.session) + \
               encode(self.target) + encode(self.q, width=4) + \
               encode(self.crc5, width=5)

    def _str_body(self) -> str:
        return (
            f"q={self.q} m={self.m.name} dr={DR.str(self.dr)} "
            f"trext={1 if self.trext else 0} session={self.session.name} "
            f"sel={self.sel.name} target={self.target.name} "
            f"crc5=0x{self.crc5:02X}"
        )


class QueryRep(Command):
    """
    Query repeat command (QueryRep).
    """
    def __init__(self, session: Session):
        """
        Constructor.

        Parameters
        ----------
        session : Session
            session generally should match the session in Query command.
        """
        self._session = session
        super().__init__(CommandCode.QUERY_REP)

    @property
    def session(self) -> Session:
        return self._session

    def _encode_body(self) -> str:
        return encode(self.session)

    def _str_body(self) -> str:
        return f"session={self._session.name}"


class Ack(Command):
    """
    Acknowledgement sent by the reader when received RN16 successfully.
    """
    def __init__(self, rn: int = 0xAAAA):
        """
        Constructor.

        Parameters
        ----------
        rn : int, optional
            random number (16 bits word) from RN16 reply. Default: 0xAAAA
            (this value has equal number of 0s and 1s)
        """
        self._rn = rn
        super().__init__(CommandCode.ACK)

    @property
    def rn(self) -> int:
        return self._rn

    def _encode_body(self) -> str:
        return encode(self.rn, width=16)

    def _str_body(self) -> str:
        return f"rn=0x{self.rn:04X}"


class ReqRn(Command):
    """
    Request another random word from the tag.

    This command is sent by the reader prior to any access command, e.g.
    Read or Write.
    """
    def __init__(self, rn: int = 0xAAAA, crc16: int = 0xAAAA):
        """
        Constructor.

        Default parameters are selected in a way they have equal number of
        0s and 1s (`0xAAAA = 1010 1010 1010 1010`).

        Parameters
        ----------
        rn : int, optional (default: 0xAAAA)
        crc16 : int, optional (default: 0xAAAA)
        """
        self._rn = rn
        self._crc16 = crc16
        super().__init__(CommandCode.REQ_RN)

    @property
    def rn(self) -> int:
        return self._rn

    @property
    def crc16(self) -> int:
        return self._crc16

    def _encode_body(self) -> str:
        return encode(self.rn, width=16) + encode(self.crc16, width=16)

    def _str_body(self) -> str:
        return f"rn=0x{self.rn:04X} crc16=0x{self.crc16:04X}"


class Read(Command):
    """
    Command to read data from the tag memory.
    """
    def __init__(self, bank: Bank = Bank.USER, wordptr: int = 0,
                 wordcnt: int = 4, rn: int = 0xAAAA, crc16: int = 0xAAAA):
        """
        Constructor.

        Parameters
        ----------
        bank : Bank, optional (default: USER)
        wordptr : int, optional (default: 0)
        wordcnt : int, optional (default: 4)
        rn : int, optional (default: 0xAAAA)
        crc16 : int, optional (default: 0xAAAA)
        """
        self._bank = bank
        self._wordptr = wordptr
        self._wordcnt = wordcnt
        self._rn = rn
        self._crc16 = crc16
        super().__init__(CommandCode.READ)

    @property
    def bank(self) -> Bank:
        return self._bank

    @property
    def wordptr(self) -> int:
        return self._wordptr

    @property
    def wordcnt(self) -> int:
        return self._wordcnt

    @property
    def rn(self) -> int:
        return self._rn

    @property
    def crc16(self) -> int:
        return self._crc16

    def _encode_body(self) -> str:
        return (encode(self.bank) + encode(self.wordptr, use_ebv=True) +
                encode(self.wordcnt, width=8) + encode(self.rn, width=16) +
                encode(self.crc16, width=16))

    def _str_body(self) -> str:
        return f"bank={self.bank.name} wordptr={self.wordptr} " \
               f"wordcnt={self.wordcnt} rn=0x{self.rn:04X} " \
               f"crc16=0x{self.crc16:04X}"


class ReaderFrame:
    """
    Reader frame joining a preamble and a command.

    The key property of the frame is `duration`. The property uses
    data-0 and data-1 durations from the preamble and counts command bits
    to compute the command duration, and adds the preamble duration to it.
    """
    def __init__(self, preamble: ReaderSync, command: Command):
        self._command = command
        self._preamble = preamble

    @property
    def msg(self):
        return self._command

    @property
    def command(self):
        return self._command

    @property
    def preamble(self):
        return self._preamble

    @cached_property
    def duration(self):
        """
        Get frame duration, incl. preamble and command durations.
        """
        nb = self.command.count_bits()
        d0, d1 = self.preamble.data0, self.preamble.data1
        return self.preamble.duration + d0 * nb[0] + d1 * nb[1]

    def __repr__(self):
        return f"(ReaderFrame: preamble={self.preamble}, "\
            "command={self.command})"
