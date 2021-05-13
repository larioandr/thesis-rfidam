import pytest
from numpy.testing import assert_allclose
import re

from rfidam.protocol.symbols import TagEncoding
from rfidam.protocol.responses import Rn16, Epc, Handle, Data, TagPreamble, \
    TagFrame


@pytest.mark.parametrize('msg, name, bitlen, encoded, string', [
    # Rn16
    (Rn16(0x0000), 'Rn16', 16, '0000000000000000', '(Rn16: rn=0x0000)'),
    (Rn16(0xFFFF), 'Rn16', 16, '1111111111111111', '(Rn16: rn=0xFFFF)'),
    # Epc
    (
        Epc('00', pc=0, crc16=0), 'Epc', 40,
        '0000000000000000000000000000000000000000',
        '(Epc: epcid=0x00 pc=0x0000 crc16=0x0000)'
    ), (
        Epc('FFFF', pc=0, crc16=0xFFFF), 'Epc', 48,
        '000000000000000011111111111111111111111111111111',
        '(Epc: epcid=0xFFFF pc=0x0000 crc16=0xFFFF)'
    ), (
        Epc([0xAB, 0xCD, 0xEF], pc=0xFFFF, crc16=0), 'Epc', 56,
        '11111111111111111010101111001101111011110000000000000000',
        '(Epc: epcid=0xABCDEF pc=0xFFFF crc16=0x0000)'
    ), (
        Epc('', pc=0, crc16=0), 'Epc', 32,
        '00000000000000000000000000000000',
        '(Epc: epcid=- pc=0x0000 crc16=0x0000)'
    ),
    # Handle
    (
        Handle(0, 0), 'Handle', 32, '00000000000000000000000000000000',
        '(Handle: rn=0x0000 crc16=0x0000)'
    ), (
        Handle(0, 0xFFFF), 'Handle', 32, '00000000000000001111111111111111',
        '(Handle: rn=0x0000 crc16=0xFFFF)'
    ), (
        Handle(0xFFFF, 0), 'Handle', 32, '11111111111111110000000000000000',
        '(Handle: rn=0xFFFF crc16=0x0000)'
    ),
    # Data
    (
        Data('', rn=0, crc16=0, header=1), 'Data', 33,
        '100000000000000000000000000000000',
        '(Data: header=1 words=- rn=0x0000 crc16=0x0000)'
    ), (
        Data('FFFFFFFF', rn=0, crc16=0xFFFF), 'Data', 65,
        '01111111111111111111111111111111100000000000000001111111111111111',
        '(Data: header=0 words=0xFFFFFFFF rn=0x0000 crc16=0xFFFF)'
    ), (
        Data([0xFF, 0x00], rn=0xFFFF, crc16=0), 'Data', 49,
        '0111111110000000011111111111111110000000000000000',
        '(Data: header=0 words=0xFF00 rn=0xFFFF crc16=0x0000)'
    )
])
def test_response_props(msg, name, bitlen, encoded, string):
    """
    Validate name, bitlen, encoded props of tag responses and repr() method.
    """
    assert msg.name == name
    assert msg.bitlen == bitlen
    assert msg.encoded == encoded
    assert str(msg) == string


@pytest.mark.parametrize('m, trext, blf, encoded, bitlen, duration', [
    (TagEncoding.FM0, False, 400e3, '1010v1', 6, 1.5e-5),
    (TagEncoding.FM0, True, 300e3, '0000000000001010v1', 18, 6e-5),
    (TagEncoding.M2, False, 100e3, '0000010111', 10, 0.0002),
    (TagEncoding.M2, True, 100e3, '0000000000000000010111', 22, 0.00044),
    (TagEncoding.M4, False, 100e3, '0000010111', 10, 0.0004),
    (TagEncoding.M8, False, 200e3, '0000010111', 10, 0.0004)
])
def test_tag_preamble(m, trext, blf, encoded, bitlen, duration):
    """
    Validate tag response preamble properties and routines.
    """
    preamble = TagPreamble(m, trext, blf)

    # Basic properties (fields)
    assert preamble.m == m
    assert_allclose(preamble.blf, blf)
    assert preamble.trext == trext

    # More advanced properties and routines (encoded string, duration, etc.)
    assert preamble.encoded == encoded
    assert preamble.bitlen == bitlen
    assert_allclose(preamble.duration, duration)

    # Check that str() method is implemented in some reasonable way:
    string = str(preamble)
    pattern = r'^\(Preamble: m=(\w+) blf=([\d\-.e]+) trext=(\w+) ' \
              r'duration=([\d\-.e]+)\)$'
    assert re.match(pattern, string)


@pytest.mark.parametrize('preamble, response, encoded, bitlen, duration', [
    (
        TagPreamble(TagEncoding.FM0, False, 400e3), Rn16(0xABCD),
        '1010v11010101111001101e', 23, 5.75e-5
    ), (
        TagPreamble(TagEncoding.M4, False, 400e3), Handle(0x1234, 0),
        '000001011100010010001101000000000000000000e', 43, 0.00043,
    )
])
def test_tag_frame(preamble, response, encoded, bitlen, duration):
    """
    Validate tag frame implementation.
    """
    frame = TagFrame(preamble, response)

    # Basic props
    assert frame.preamble is preamble
    assert frame.msg is response

    # Advanced props
    assert frame.encoded == encoded
    assert frame.bitlen == bitlen
    assert_allclose(frame.duration, duration)

    # Check that TagFrame.str() is implemented:
    string = str(frame)
    pattern = r'^\(TagFrame: preamble=\(Preamble: ([\w\s\-:.=]+)\) '\
              r'msg=\(([\w\s\-:.=]+)\) duration=([\d\-e.]+)\)$'
    assert re.match(pattern, string)
