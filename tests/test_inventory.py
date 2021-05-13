import pytest
from numpy.testing import assert_allclose
import numpy as np

from rfidam.protocol.symbols import TagEncoding, DR
from rfidam import inventory
from rfidam.protocol.protocol import Protocol, LinkProps


PROTOCOL_SETTINGS = {
    'tari': 12.5e-6,
    'rtcal': 37.5e-6,
    'trcal': 50e-6,
    'm': TagEncoding.M2,
    'dr': DR.DR_643,
    'trext': False,
    'q': 4,
    't_off': 0.1,
    'n_epcid_bytes': 12,
    'n_data_words': 4
}

# Since Protocol instances are immutable, define them here and use everywhere:
PROTO_EPC = Protocol(LinkProps(**PROTOCOL_SETTINGS, use_tid=False))
PROTO_TID = Protocol(LinkProps(**PROTOCOL_SETTINGS, use_tid=True))


#
# TEST SLOT DURATIONS
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    'protocol, n_tags, ber, t_empty, t_reply, t_collided', [
        # EPCID-only protocol:
        (PROTO_EPC, 0, 0.001, 0.000075, np.nan, 0.000191),
        (PROTO_EPC, 2, 0.001, 0.000075, 0.001288, 0.000191),
        (PROTO_EPC, 20, 0.001, 0.000075, 0.001290, 0.000191),
        (PROTO_EPC, 0, 0.02, 0.000075, np.nan, 0.000191),
        (PROTO_EPC, 2, 0.02, 0.000075, 0.001000, 0.000191),
        (PROTO_EPC, 20, 0.02, 0.000075, 0.000999, 0.000191),
        # EPCID + TID:
        (PROTO_TID, 0, 0.001, 0.000075, np.nan, 0.000191),
        (PROTO_TID, 2, 0.001, 0.000075, 0.003573, 0.000191),
        (PROTO_TID, 20, 0.001, 0.000075, 0.003571, 0.000191),
        (PROTO_TID, 0, 0.02, 0.000075, np.nan, 0.000191),
        (PROTO_TID, 2, 0.02, 0.000075, 0.001098, 0.000191),
        (PROTO_TID, 20, 0.02, 0.000075, 0.001102, 0.000191),
    ]
)
def test_slot_model__durations(protocol, n_tags, ber, t_empty, t_reply,
                               t_collided):
    """Validate empty slot duration model."""
    model = inventory.create_slot_model(protocol, n_tags=n_tags, ber=ber)
    assert_allclose(model.durations.empty, t_empty, rtol=.01, atol=1e-6)
    if t_reply is not np.nan:
        assert_allclose(model.durations.reply, t_reply, rtol=.01, atol=1e-6)
    assert_allclose(model.durations.collided, t_collided, rtol=.01, atol=1e-6)


@pytest.mark.parametrize(
    'protocol, n_tags, ber, p_empty, p_reply, p_collided', [
        (PROTO_EPC, 0, 0.001, 1.000000, 0.000000, 0.000000),
        (PROTO_EPC, 2, 0.001, 0.879012, 0.116975, 0.004012),
        (PROTO_EPC, 20, 0.001, 0.275281, 0.366206, 0.358513),
        (PROTO_EPC, 0, 0.02, 1.000000, 0.000000, 0.000000),
        (PROTO_EPC, 2, 0.02, 0.879106, 0.116788, 0.004106),
        (PROTO_EPC, 20, 0.02, 0.276587, 0.364794, 0.358619),
        (PROTO_TID, 0, 0.001, 1.000000, 0.000000, 0.000000),
        (PROTO_TID, 2, 0.001, 0.878950, 0.117100, 0.003950),
        (PROTO_TID, 20, 0.001, 0.276350, 0.365356, 0.358294),
        (PROTO_TID, 0, 0.02, 1.000000, 0.000000, 0.000000),
        (PROTO_TID, 2, 0.02, 0.878900, 0.117200, 0.003900),
        (PROTO_TID, 20, 0.02, 0.273863, 0.367681, 0.358456),
    ]
)
def test_slot_model__probs(protocol, n_tags, ber, p_empty, p_reply,
                           p_collided):
    """Validate slot probabilities estimations.
    """
    model = inventory.create_slot_model(protocol, n_tags=n_tags, ber=ber)
    assert_allclose(model.probs.empty, p_empty, rtol=.1, atol=1e-6)
    assert_allclose(model.probs.reply, p_reply, rtol=.1, atol=1e-6)
    assert_allclose(model.probs.collided, p_collided, rtol=.1, atol=1e-6)


#
# TEST ROUND DURATIONS
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('protocol, n_tags, ber, expected_duration', [
    (PROTO_EPC, 0, 0.001, 0.003638),
    (PROTO_EPC, 2, 0.001, 0.005922),
    (PROTO_EPC, 20, 0.001, 0.011366),
    (PROTO_EPC, 0, 0.02, 0.003638),
    (PROTO_EPC, 2, 0.02, 0.005381),
    (PROTO_EPC, 20, 0.02, 0.009745),
    (PROTO_TID, 0, 0.001, 0.003638),
    (PROTO_TID, 2, 0.001, 0.010188),
    (PROTO_TID, 20, 0.001, 0.024780),
    (PROTO_TID, 0, 0.02, 0.003638),
    (PROTO_TID, 2, 0.02, 0.005553),
    (PROTO_TID, 20, 0.02, 0.010363),
])
def test_round_model__durations(protocol, n_tags, ber, expected_duration):
    """Validate round duration estimation.
    """
    model = inventory.create_round_model(protocol, n_tags=n_tags, ber=ber)
    assert_allclose(model.round_duration, expected_duration, rtol=.01)
