from collections.abc import Iterable
from typing import Optional, Union, Sequence
from functools import cached_property

from rfidam.protocol.symbols import encode, TagEncoding

DEFAULT_EPC = 'A5' * 12


class Response:
    """
    Base class for tag responses (replies).

    Any derived class must implement two methods:

    - _encode()
    - _str_body()

    All responses should be treated as immutable objects. For performance
    reasons, `encoded` and `bitlen` properties are cached, so if fields
    are updated, these properties caches should be invalidated.
    """
    @cached_property
    def encoded(self) -> str:
        """
        Get a response as a bit-encoded string.
        """
        return self._encode()

    @cached_property
    def bitlen(self) -> int:
        """
        Get number of bits in the encoded response.
        """
        return len(self.encoded)

    @property
    def name(self) -> str:
        """
        Get response human-readable name.
        """
        return self.__class__.__name__

    def _encode(self) -> str:
        """
        Abstract method that should return bit-string of encoded response.
        """
        raise NotImplementedError()

    def _str_body(self) -> str:
        """
        Abstract method that should return a human-readable response string.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """
        Get string representation of the response.
        """
        return f'({self.name}: {self._str_body()})'


class Rn16(Response):
    """
    A response that tag should send after QueryRep when counter reaches zero.

    Contains only random 16-bit word (RN).
    """
    def __init__(self, rn: int = 0xAAAA):
        """
        Constructor.

        Parameters
        ----------
        rn : int
            a random word (16 bits) between 0x0000 and 0xFFFF.
        """
        self._rn = rn

    @property
    def rn(self) -> int:
        return self._rn

    def _encode(self) -> str:
        return encode(self.rn, width=16)

    def _str_body(self) -> str:
        return f"rn=0x{self.rn:04X}"


class Epc(Response):
    """
    A response on Ack commmand with EPCID, PC and CRC-16.
    """
    def __init__(self, epcid: Union[str, bytes, Sequence[int]] = DEFAULT_EPC,
                 pc: int = 0x0000, crc16: int = 0x0000):
        """
        Constructor.

        Parameters
        ----------
        epcid : str, bytes or a sequence of ints, optional
            EPCID of the tag. Can be given with a string (e.g. 'ABCD'), bytes
            or a sequence of integers representing bytes.
            By default, DEFAULT_EPC.
        pc : int
            PC word between 0x0000 and 0xFFFF (default: 0)
        crc16 : int
            checksum word between 0x0000 and 0xFFFF (default: 0)
        """
        self._epcid = epcid
        self._pc = pc
        self._crc16 = crc16

    @property
    def epcid(self) -> str:
        return self._epcid

    @property
    def pc(self) -> int:
        return self._pc

    @property
    def crc16(self) -> int:
        return self._crc16

    def _encode(self) -> str:
        return encode(self.pc, width=16) + encode(self.epcid) + \
               encode(self.crc16, width=16)

    def _str_body(self) -> str:
        return f"epcid={_str_bitstring(self.epcid)} " \
               f"pc=0x{self.pc:04X} crc16=0x{self.crc16:04X}"


class Handle(Response):
    """
    A response on ReqRn command.
    """
    def __init__(self, rn: int = 0xAAAA, crc16: int = 0x0000):
        """
        Constructor.

        Parameters
        ----------
        rn : int, optional
            a random word between 0 and 0xFFFF (default: 0xAAAA)
        crc16 : int, optional
            a 16-bit checksum (default: 0)
        """
        self._rn = rn
        self._crc16 = crc16

    @property
    def rn(self) -> int:
        return self._rn

    @property
    def crc16(self) -> int:
        return self._crc16

    def _encode(self) -> str:
        return encode(self.rn, width=16) + encode(self.crc16, width=16)

    def _str_body(self) -> str:
        return f"rn=0x{self.rn:04X} crc16=0x{self.crc16:04X}"


class Data(Response):
    """
    A response on Read command.
    """
    def __init__(self, words: Union[str, bytes, Sequence[int]] = 'ABCD' * 4,
                 rn: int = 0, crc16: int = 0, header: int = 0):
        """
        Constructor.

        Parameters
        ----------
        words : str, bytes or sequence of ints, optional
            data that should be returned to the reader
            (default: 'ABCDABCDABCDABCD')
        rn : int, optional
            random word (default: 0)
        crc16 : int, optional
            16-bits checksum (default: 0)
        header : int, optional
            int, should always be set to zero (default)
        """
        self._header = header
        self._words = words
        self._rn = rn
        self._crc16 = crc16

    @property
    def words(self) -> str:
        return self._words

    @property
    def header(self) -> int:
        return self._header

    @property
    def rn(self) -> int:
        return self._rn

    @property
    def crc16(self) -> int:
        return self._crc16

    def _encode(self) -> str:
        return encode(self.header, width=1) + encode(self.words) + \
               encode(self.rn, width=16) + \
               encode(self.crc16, width=16)

    def _str_body(self) -> str:
        return f"header={self.header:01X} " \
               f"words={_str_bitstring(self.words)} " \
               f"rn=0x{self.rn:04X} crc16=0x{self.crc16:04X}"


class TagPreamble:
    """
    Tag preamble.

    Preamble is characterized with M (tag encoding), BLF and TRext flag.
    It has a bit-string representation ('v' is used for invalid symbol)
    and has duration (in seconds).

    Preamble is assumed to be immutable, its `duration`, `bitlen` and
    `encoded` props are cached.
    """
    def __init__(self, m: TagEncoding, trext: bool, blf: float):
        """
        Constructor.

        Parameters
        ----------
        m : TagEncoding
        trext : bool
        blf : float
        """
        self._m = m
        self._trext = trext
        self._blf = blf

    @property
    def m(self) -> TagEncoding:
        """
        Tag encoding.
        """
        return self._m

    @property
    def trext(self) -> bool:
        """
        Flag indicating whether the preamble is extended.
        """
        return self._trext

    @property
    def blf(self) -> float:
        """
        Get backscatter link frequency in Hz.
        """
        return self._blf

    @cached_property
    def encoded(self) -> str:
        """
        Get bit-encoded string representation of the preamble.

        If the preamble contains invalid symbol (case for m=FM0), it is
        written as 'v'.
        """
        if self.m == TagEncoding.FM0:
            return '1010v1' if not self.trext else '0000000000001010v1'
        if self.trext:
            return '0000000000000000010111'
        return '0000010111'

    @cached_property
    def bitlen(self) -> int:
        """
        Get preamble size in bits.
        """
        return len(self.encoded)

    @cached_property
    def duration(self) -> float:
        """
        Get preamble duration in seconds.
        """
        return self.bitlen * (self.m.value / self.blf)

    def __str__(self) -> str:
        return f'(Preamble: m={TagEncoding.str(self.m)} blf={self.blf:g} '\
               f'trext={self.trext} duration={self.duration:g})'


class TagFrame:
    """
    Tag frame implementation.

    Frame contains preamble and message (response). Using a preamble,
    frame can compute its duration and encoded string.

    TagFrame is assumed to be immutable, so props like `bitlen`,
    `encoded` or `duration` are cached.
    """
    def __init__(self, preamble: TagPreamble, response: Response):
        """
        Constructor.

        Parameters
        ----------
        preamble : TagPreamble
        response : Response
        """
        self._preamble = preamble
        self._message = response

    @property
    def preamble(self) -> TagPreamble:
        """
        Get tag preamble.
        """
        return self._preamble

    @property
    def msg(self) -> Response:
        """
        Get tag response (message).
        """
        return self._message

    @cached_property
    def encoded(self) -> str:
        """
        Get the frame representation in a bit string.
        """
        return f'{self._preamble.encoded}{self._message.encoded}e'

    @cached_property
    def bitlen(self) -> int:
        """
        Get number of bits in the encoded bit string.
        """
        return len(self.encoded)

    @cached_property
    def duration(self) -> float:
        """
        Get frame duration in seconds.
        """
        return self.bitlen * (self.preamble.m.value / self.preamble.blf)

    def __str__(self) -> str:
        return f'(TagFrame: preamble={self.preamble} msg={self.msg} '\
               f'duration={self.duration:g})'


def _str_bitstring(data: Optional[Union[str, Iterable]], empty='-'):
    if not isinstance(data, str):
        data = "".join([format(x, '02X') for x in data])
    return empty if not data else '0x' + data
