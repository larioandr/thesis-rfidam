from enum import Enum
import numpy as np
import binascii
from collections.abc import Iterable


class DR(Enum):
    """Division ratio. Takes values 8 or 64/3.
    """
    DR_8 = 0
    DR_643 = 1

    @staticmethod
    def encode(dr) -> str:
        """Get 01-encoded string for the symbol.
        """
        return str(dr.value)

    @property
    def ratio(self) -> float:
        """Get the ratio in float format (8.0 or 64/3)
        """
        return 8.0 if self == DR.DR_8 else 64.0 / 3

    @staticmethod
    def parse(s: str) -> 'DR':
        """Parse a string to get the DR symbol.
        """
        if s == '8':
            return DR.DR_8
        elif s == '64/3':
            return DR.DR_643
        raise ValueError(f'unrecognized DR "{s}"')

    @staticmethod
    def str(value: 'DR') -> str:
        """Convert a symbol to a string.
        """
        if value == DR.DR_8:
            return '8'
        if value == DR.DR_643:
            return '64/3'
        raise ValueError(f'unrecognized DR "{value}"')


class TagEncoding(Enum):
    """Tag encoding (FM0, M2, M4 or M8).
    """
    FM0 = 1
    M2 = 2
    M4 = 4
    M8 = 8

    @staticmethod
    def encode(m: 'TagEncoding') -> str:
        """Get 01-encoded string (length = 2) for the symbol.
        """
        return format(int(np.log2(m.value)), '02b')

    @staticmethod
    def parse(s: str) -> 'TagEncoding':
        """Parse a string to get the TagEncoding symbol.
        """
        s = str(s).upper()
        if s in {'1', 'FM0'}:
            return TagEncoding.FM0
        elif s in {'2', 'M2'}:
            return TagEncoding.M2
        elif s in {'4', 'M4'}:
            return TagEncoding.M4
        elif s in {'8', 'M8'}:
            return TagEncoding.M8
        raise ValueError(f'unrecognized TagEncoding = "{s}"')

    @staticmethod
    def str(value: 'TagEncoding') -> str:
        """Convert a symbol to a string.
        """
        if value == TagEncoding.FM0:
            return 'FM0'
        if value == TagEncoding.M2:
            return 'M2'
        if value == TagEncoding.M4:
            return 'M4'
        if value == TagEncoding.M8:
            return 'M8'
        raise ValueError(f'unrecognized TagEncoding "{value}"')


class Bank(Enum):
    """Memory bank (RESERVED, EPC, TID or USER).
    """
    RESERVED = 0
    EPC = 1
    TID = 2
    USER = 3

    @staticmethod
    def encode(bank: 'Bank') -> str:
        """Get 01-encoded string (length = 2) for the symbol.
        """
        return format(bank.value, '02b')

    @staticmethod
    def parse(s: str) -> 'Bank':
        """Parse a string to get the symbol.
        """
        s = s.upper()
        if s == 'RESERVED':
            return Bank.RESERVED
        elif s == 'EPC':
            return Bank.EPC
        elif s == 'TID':
            return Bank.TID
        elif s == 'USER':
            return Bank.USER
        raise ValueError(f'unrecognized Bank = "{s}"')

    @staticmethod
    def str(value: 'Bank') -> str:
        """Convert a symbol to a string.
        """
        if value == Bank.RESERVED:
            return 'RESERVED'
        if value == Bank.EPC:
            return 'EPC'
        if value == Bank.TID:
            return 'TID'
        if value == Bank.USER:
            return 'USER'
        raise ValueError(f'unrecognized Bank "{value}"')


class InventoryFlag(Enum):
    """Inventory flag that tag stores for each session (A or B)
    """
    A = 0
    B = 1

    @staticmethod
    def encode(flag: 'InventoryFlag') -> str:
        """Get 01-encoded string (1 bit) for the symbol. We assume A = 0.
        """
        return str(flag.value)

    def invert(self) -> 'InventoryFlag':
        """Get the inversion of the flag (A <-> B).
        """
        return InventoryFlag.A if self == InventoryFlag.B else InventoryFlag.B

    @staticmethod
    def parse(s: str) -> 'InventoryFlag':
        """Parse a string to get the symbol.
        """
        s = s.upper()
        if s == 'A':
            return InventoryFlag.A
        elif s == 'B':
            return InventoryFlag.B
        raise ValueError(f'unrecognized InventoryFlag = "{s}"')

    @staticmethod
    def str(value: 'InventoryFlag') -> str:
        """Convert a symbol to a string.
        """
        if value == InventoryFlag.A:
            return 'A'
        if value == InventoryFlag.B:
            return 'B'
        raise ValueError(f'unrecognized InventoryFlag "{value}"')


class Sel(Enum):
    """Sel flag (ALL, SEL, ~SEL).
    """
    ALL = 0
    NO = 2
    YES = 3

    @staticmethod
    def encode(field: 'Sel') -> str:
        """Get 01-encoded string (2 bits) for the symbol.
        """
        return format(field.value, '02b')

    @staticmethod
    def parse(s: str) -> 'Sel':
        """Parse a string to get the symbol.
        """
        s = s.upper()
        if s == 'ALL':
            return Sel.ALL
        elif s == 'YES' or s == 'SEL':
            return Sel.YES
        elif s == 'NO' or s == '~SEL':
            return Sel.NO
        raise ValueError(f'unrecognized Sel = "{s}"')

    @staticmethod
    def str(value: 'Sel') -> str:
        """Convert a symbol to a string.
        """
        if value == Sel.ALL:
            return 'ALL'
        if value == Sel.NO:
            return 'NO'
        if value == Sel.YES:
            return 'YES'
        raise ValueError(f'unrecognized Sel "{value}"')


class Session(Enum):
    """Session flag (S0, S1, S2, S3).
    """
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3

    @staticmethod
    def encode(session: 'Session') -> str:
        """Get 01-encoded string (2 bits) for the symbol.
        """
        return format(session.value, '02b')

    @staticmethod
    def parse(s: str) -> 'Session':
        """Parse a string to get the symbol.
        """
        s = str(s).upper()
        if s == '0' or s == 'S0':
            return Session.S0
        elif s == '1' or s == 'S1':
            return Session.S1
        elif s == '2' or s == 'S2':
            return Session.S2
        elif s == '3' or s == 'S3':
            return Session.S3
        raise ValueError(f'unrecognized Session = "{s}"')

    @staticmethod
    def str(value: 'Session') -> str:
        """Convert a symbol to a string.
        """
        if value == Session.S0:
            return 'S0'
        if value == Session.S1:
            return 'S1'
        if value == Session.S2:
            return 'S2'
        if value == Session.S3:
            return 'S3'
        raise ValueError(f'unrecognized Session "{value}"')


def encode_ebv(value):
    """
    Get 01-encoded string for integer value in EBV encoding.

    Parameters
    ----------
    value : int
    """
    def _encode(value_, first_block):
        prefix = '0' if first_block else '1'
        if value_ < 128:
            return prefix + format(value_, '07b')
        return _encode(value_ >> 7, first_block=False) + \
            _encode(value_ % 128, first_block=first_block)

    return _encode(value, first_block=True)


class CommandCode(Enum):
    QUERY = 16
    QUERY_REP = 0
    ACK = 1
    REQ_RN = 193
    READ = 194

    @staticmethod
    def encode(code):
        if code == CommandCode.QUERY:
            return '1000'
        if code == CommandCode.QUERY_REP:
            return '00'
        if code == CommandCode.ACK:
            return '01'
        if code == CommandCode.REQ_RN:
            return '11000001'
        if code == CommandCode.READ:
            return '11000010'
        raise ValueError(f'unsupported command code "{code}"')

    @staticmethod
    def get_name_for(code):
        if code == CommandCode.QUERY:
            return 'Query'
        if code == CommandCode.QUERY_REP:
            return 'QueryRep'
        if code == CommandCode.ACK:
            return 'Ack'
        if code == CommandCode.REQ_RN:
            return 'ReqRn'
        if code == CommandCode.READ:
            return 'Read'
        raise ValueError(f'unsupported command code "{code}"')


def encode(value, width=0, use_ebv=False):
    tv = type(value)
    if tv in {DR, TagEncoding, Bank, InventoryFlag, Sel, Session, CommandCode}:
        return tv.encode(value)

    if tv == bool:
        return '1' if value else '0'

    elif tv == int:
        if use_ebv:
            return encode_ebv(value)
        elif width > 0:
            return format(value, "0{width}b".format(width=width))
        return format(value, "b")

    elif tv == str:
        return encode(list(binascii.unhexlify(value.strip())))

    elif tv == bytes:
        return encode(list(value))

    elif isinstance(value, Iterable):
        return "".join(format(x, "08b") for x in value)

    raise ValueError(f'unsupported field type "{tv}"')


def min_t1(rtcal, blf, frt=0.1):
    return max(rtcal, 10.0 / blf) * (1. - frt) - 2e-6


def nominal_t1(rtcal, blf):
    return max(rtcal, 10 / blf)


def max_t1(rtcal, blf, frt=0.1):
    return max(rtcal, 10.0 / blf) * (1. + frt) + 2e-6


def min_t2(blf):
    return 3. / blf


def max_t2(blf):
    return 20. / blf


def t3():
    return 1e-6


def t4(rtcal):
    return 2. * rtcal


def get_blf(dr, trcal):
    return dr.ratio / trcal
