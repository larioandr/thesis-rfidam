import pytest
from numpy.testing import assert_allclose

from rfidam.protocol.commands import ReaderSync, ReaderPreamble, Query, \
    QueryRep, Ack, ReqRn, Read, ReaderFrame
from rfidam.protocol.symbols import TagEncoding, DR, Sel, Session, \
    InventoryFlag, Bank


@pytest.mark.parametrize('delim, tari, rtcal, data1, duration, string', [
    (
        12.5e-6, 6.25e-6, 18.75e-6, 12.5e-6, 37.5e-6,
        '(Sync: delim=12.5us tari=6.25us rtcal=18.75us)'
    ), (
        20e-6, 25e-6, 60e-6, 35e-6, 105e-6,
        '(Sync: delim=20us tari=25us rtcal=60us)'
    )
])
def test_sync(delim, tari, rtcal, data1, duration, string):
    """
    Validate ReaderSync properties.
    """
    sync = ReaderSync(tari, rtcal, delim)
    assert_allclose(sync.tari, tari)
    assert_allclose(sync.rtcal, rtcal)
    assert_allclose(sync.data0, tari)
    assert_allclose(sync.data1, data1)
    assert_allclose(sync.duration, duration)
    assert str(sync) == string


@pytest.mark.parametrize('delim, tari, rtcal, trcal, string', [
    (
        12.5e-6, 6.25e-6, 18.75e-6, 35.5e-6,
        '(Preamble: delim=12.5us tari=6.25us rtcal=18.75us trcal=35.5us)'
    ), (
        20e-6, 25e-6, 60e-6, 100e-6,
        '(Preamble: delim=20us tari=25us rtcal=60us trcal=100us)'
    )
])
def test_preamble(delim, tari, rtcal, trcal, string):
    """
    Validate ReaderPreamble properties.
    """
    preamble = ReaderPreamble(tari, rtcal, trcal, delim)
    assert_allclose(preamble.tari, tari)
    assert_allclose(preamble.rtcal, rtcal)
    assert_allclose(preamble.trcal, trcal)
    assert_allclose(preamble.data0, tari)
    assert_allclose(preamble.data1, rtcal - tari)
    assert_allclose(preamble.duration, delim + tari + rtcal + trcal)
    assert str(preamble) == string


@pytest.mark.parametrize('cmd, encoded, bits_count, bitlen, string', [
    # Query
    (
        Query(0, TagEncoding.FM0, DR.DR_8, False, Sel.ALL, Session.S0,
              InventoryFlag.A, crc5=0),
        '1000000000000000000000', (21, 1), 22,
        '(Query: q=0 m=FM0 dr=8 trext=0 session=S0 sel=ALL target=A crc5=0x00)'
    ), (
        Query(15, TagEncoding.FM0, DR.DR_643, True, Sel.ALL, Session.S3,
              InventoryFlag.A, crc5=0),
        '1000100100110111100000', (13, 9), 22,
        '(Query: q=15 m=FM0 dr=64/3 trext=1 session=S3 sel=ALL target=A '
        'crc5=0x00)'
    ), (
        Query(0, TagEncoding.M8, DR.DR_8, False, Sel.YES, Session.S0,
              InventoryFlag.B, crc5=0x1F),
        '1000011011001000011111', (11, 11), 22,
        '(Query: q=0 m=M8 dr=8 trext=0 session=S0 sel=YES target=B crc5=0x1F)'
    ), (
        Query(15, TagEncoding.M8, DR.DR_643, True, Sel.YES, Session.S3,
              InventoryFlag.B, crc5=0x1F),
        '1000111111111111111111', (3, 19), 22,
        '(Query: q=15 m=M8 dr=64/3 trext=1 session=S3 sel=YES target=B '
        'crc5=0x1F)'
    ),
    # QueryRep
    (QueryRep(Session.S0), '0000', (4, 0), 4, '(QueryRep: session=S0)'),
    (QueryRep(Session.S3), '0011', (2, 2), 4, '(QueryRep: session=S3)'),
    # Ack
    (Ack(0x0000), '010000000000000000', (17, 1), 18, '(Ack: rn=0x0000)'),
    (Ack(0xFFFF), '011111111111111111', (1, 17), 18, '(Ack: rn=0xFFFF)'),
    # ReqRn
    (
        ReqRn(0x0000, 0x0000), '1100000100000000000000000000000000000000',
        (37, 3), 40, '(ReqRn: rn=0x0000 crc16=0x0000)'
    ), (
        ReqRn(0xFFFF, 0xFFFF), '1100000111111111111111111111111111111111',
        (5, 35), 40, '(ReqRn: rn=0xFFFF crc16=0xFFFF)'
    ),
    # Read
    (
        Read(Bank.RESERVED, 0, 0, 0x0000, crc16=0x0000),
        '1100001000000000000000000000000000000000000000000000000000',
        (55, 3), 58,
        '(Read: bank=RESERVED wordptr=0 wordcnt=0 rn=0x0000 crc16=0x0000)'
    ), (
        Read(Bank.RESERVED, 16383, 0, 0x0000, crc16=0x0000),
        '110000100011111111011111110000000000000000000000000000000000000000',
        (48, 18), 66,
        '(Read: bank=RESERVED wordptr=16383 wordcnt=0 rn=0x0000 crc16=0x0000)'
    ), (
        Read(Bank.RESERVED, 16384, 0, 0x0000, crc16=0x0000),
        '11000010001000000110000000000000000000000000000000000000000000000000'
        '000000',
        (68, 6), 74,
        '(Read: bank=RESERVED wordptr=16384 wordcnt=0 rn=0x0000 crc16=0x0000)'
    ), (
        Read(Bank.USER, 0, 255, 0x0000, crc16=0xFFFF),
        '1100001011000000001111111100000000000000001111111111111111',
        (29, 29), 58,
        '(Read: bank=USER wordptr=0 wordcnt=255 rn=0x0000 crc16=0xFFFF)'
    ), (
        Read(Bank.RESERVED, 127, 0, 0xFFFF, crc16=0x0000),
        '1100001000011111110000000011111111111111110000000000000000',
        (32, 26), 58,
        '(Read: bank=RESERVED wordptr=127 wordcnt=0 rn=0xFFFF crc16=0x0000)'
    ), (
        Read(Bank.USER, 127, 255, 0xFFFF, crc16=0xFFFF),
        '1100001011011111111111111111111111111111111111111111111111',
        (6, 52), 58,
        '(Read: bank=USER wordptr=127 wordcnt=255 rn=0xFFFF crc16=0xFFFF)'
    )
])
def test_commands_props(cmd, encoded, bits_count, bitlen, string):
    """
    Validate commands properties.
    """
    assert cmd.encoded == encoded
    assert cmd.count_bits() == bits_count
    assert cmd.bitlen == bitlen
    assert str(cmd) == string


@pytest.mark.parametrize('preamble, command', [(
    ReaderPreamble(6.25e-6, 18.75e-6, 37.5e-6),
    Query(0, TagEncoding.FM0, DR.DR_8, False, Sel.ALL, Session.S0,
          InventoryFlag.A, crc5=0)
), (
    ReaderSync(12.5e-6, 25e-6), QueryRep(Session.S3)
)])
def test_frame_props(preamble, command):
    """
    Validate frame duration estimation.
    """
    d0, d1 = preamble.data0, preamble.data1
    bits_count = command.count_bits()
    command_duration = bits_count[0] * d0 + bits_count[1] * d1

    frame = ReaderFrame(preamble, command)

    assert frame.preamble == preamble
    assert frame.msg == command

    assert_allclose(frame.duration, preamble.duration + command_duration)
