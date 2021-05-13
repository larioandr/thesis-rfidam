import pytest
from numpy.testing import assert_allclose

from rfidam.protocol.symbols import DR, TagEncoding, Bank, InventoryFlag, \
    Sel, Session, encode_ebv, CommandCode, encode, min_t1, max_t1, \
    nominal_t1, min_t2, max_t2, t3, t4, get_blf


@pytest.mark.parametrize('klass, symbol, code, strings', [
    # DR
    (DR, DR.DR_8, '0', ['8']),
    (DR, DR.DR_643, '1', ['64/3']),
    # TagEncoding
    (TagEncoding, TagEncoding.FM0, '00', ['FM0', '1']),
    (TagEncoding, TagEncoding.M2, '01', ['M2', '2']),
    (TagEncoding, TagEncoding.M4, '10', ['M4', '4']),
    (TagEncoding, TagEncoding.M8, '11', ['M8', '8']),
    # Bank
    (Bank, Bank.RESERVED, '00', ['RESERVED']),
    (Bank, Bank.EPC, '01', ['EPC']),
    (Bank, Bank.TID, '10', ['TID']),
    (Bank, Bank.USER, '11', ['USER']),
    # InventoryFlag
    (InventoryFlag, InventoryFlag.A, '0', ['A']),
    (InventoryFlag, InventoryFlag.B, '1', ['B']),
    # Sel
    (Sel, Sel.ALL, '00', ['ALL']),
    (Sel, Sel.NO, '10', ['NO', '~SEL']),
    (Sel, Sel.YES, '11', ['YES', 'SEL']),
    # Session
    (Session, Session.S0, '00', ['S0', '0']),
    (Session, Session.S1, '01', ['S1', '1']),
    (Session, Session.S2, '10', ['S2', '2']),
    (Session, Session.S3, '11', ['S3', '3']),
])
def test_symbol_props(klass, symbol, code, strings):
    """
    Validate symbol properties. The first element of strings is expected
    to be returned by `klass.str(symbol)` call.
    """
    assert getattr(klass, 'encode')(symbol) == code
    assert getattr(klass, 'str')(symbol) == strings[0]
    for string in strings:
        assert getattr(klass, 'parse')(string) == symbol


@pytest.mark.parametrize('dr, ratio', [(DR.DR_8, 8.0), (DR.DR_643, 64/3)])
def test_dr__ratio(dr, ratio):
    """Validate DR ratio property.
    """
    assert dr.ratio == ratio


def test_inventory_flag__invert():
    """Validate InventoryFlag.invert() method.
    """
    assert InventoryFlag.A.invert() == InventoryFlag.B
    assert InventoryFlag.B.invert() == InventoryFlag.A


@pytest.mark.parametrize('value, expected', [
    (0, '00000000'),
    (1, '00000001'),
    (127, '01111111'),
    (128, '1000000100000000'),
    (16383, '1111111101111111'),
    (16384, '100000011000000000000000'),
])
def test_encode_ebv(value, expected):
    """Validate encode_ebv() routine.
    """
    assert encode_ebv(value) == expected


@pytest.mark.parametrize('cmd_code, code, name', [
    (CommandCode.QUERY, '1000', 'Query'),
    (CommandCode.QUERY_REP, '00', 'QueryRep'),
    (CommandCode.ACK, '01', 'Ack'),
    (CommandCode.REQ_RN, '11000001', 'ReqRn'),
    (CommandCode.READ, '11000010', 'Read'),
])
def test_command_code(cmd_code, code, name):
    """Validate CommandCode properties.
    """
    assert CommandCode.encode(cmd_code) == code
    assert CommandCode.get_name_for(cmd_code) == name


@pytest.mark.parametrize('value, string', [
    # Make sure that encoding enum symbols just call static `encode()`
    (DR.DR_643, DR.encode(DR.DR_643)),
    (TagEncoding.M4, TagEncoding.encode(TagEncoding.M4)),
    (Bank.TID, Bank.encode(Bank.TID)),
    # Make sure that integers are encoded with EBV
    (127, '01111111'),
    (16384, '100000011000000000000000'),
    # Make sure that string is treated as hex-string and encoded:
    ('01234567', ''.join(['00000001', '00100011', '01000101', '01100111'])),
    # Make sure that byte-strings are also encoded properly:
    (bytes([0xAB, 0xCD, 0xEF]), ''.join(['10101011', '11001101', '11101111'])),
    # Validate that arrays are treated as byte sequences and encoded properly:
    ([0x1A, 0x2F, 0x9D], ''.join(['00011010', '00101111', '10011101'])),
])
def test_encode(value, string):
    """Validate that encode() routine accepts various datatypes and encode
    the values properly
    """
    assert encode(value, use_ebv=True) == string


@pytest.mark.parametrize('value, width, string', [
    (128, 0, '10000000'),     # if width=0, as many bits is used as needed
    (0x7, 0, '111'),          # the same thing as above
    (128, 10, '0010000000'),  # prefix with zeros if width > required
    (128, 5, '10000000'),     # ignore width if widht < required
])
def test_encode_int__non_ebv_mode(value, width, string):
    """Validate that encode(value, width=N, use_ebv=False) encodes integer
    in non-EBV mode with a given width.
    """
    assert encode(value, width=width, use_ebv=False) == string


@pytest.mark.parametrize('rtcal, blf, frt, expected_min, expected_max', [
    (20e-6, 300e3, .1, 2.8e-5, 3.8666667e-5),
    (100e-6, 107e3, .05, 9.3e-5, 0.000107),
])
def test_t1(rtcal, blf, frt, expected_min, expected_max):
    """Validate min_t1(), max_t1() and nominal_t1().
    """
    assert_allclose(min_t1(rtcal, blf, frt), expected_min)
    assert_allclose(max_t1(rtcal, blf, frt), expected_max)
    assert expected_min < nominal_t1(rtcal, blf) < expected_max


@pytest.mark.parametrize('blf, expected_min, expected_max', [
    (1e5, 3e-5, 2e-4), (3e5, 1e-5, 6.666667e-5)
])
def test_t2(blf, expected_min, expected_max):
    """Validate min_t2() and max_t2().
    """
    assert_allclose(min_t2(blf), expected_min)
    assert_allclose(max_t2(blf), expected_max)


def test_t3():
    """Validate that t3() returns some non-negative value.
    """
    assert t3() > 0


@pytest.mark.parametrize('rtcal, expected', [(2e-5, 4e-5), (1e-4, 2e-4)])
def test_t4(rtcal, expected):
    """Validate t4() routine.
    """
    assert_allclose(t4(rtcal), expected)


@pytest.mark.parametrize('dr, trcal, expected', [
    (DR.DR_643, 33.3e-6, 640e3),
    (DR.DR_643, 83.3e-6, 256e3),
    (DR.DR_8, 25e-6, 320e3),
    (DR.DR_8, 50e-6, 160e3)
])
def test_get_blf(dr, trcal, expected):
    """Validate get_blf() routine.
    """
    assert_allclose(get_blf(dr, trcal), expected, rtol=.01)
