from pprint import pprint

from rfidam.protocol.symbols import TagEncoding, DR
from rfidam.scenario import parse_scenario
from rfidam.simulation import ModelParams, build_scenario_info
from rfidam.protocol.protocol import Protocol, LinkProps
from rfidam.simulation import simulate


def main():
    props = LinkProps(
        tari=12.5e-6,
        rtcal=25e-6,
        trcal=40e-6,
        m=TagEncoding.M2,
        dr=DR.DR_643,
        trext=False,
        q=2,
        use_tid=False)
    protocol = Protocol(props)
    params = ModelParams(
        protocol,
        arrivals=list(range(5000)),
        time_in_area=2.3,
        scenario=parse_scenario("ABABx"),
        ber=.02)

    journal = simulate(params)
    sc_info = build_scenario_info([journal], 4)
    print("\nRounds durations:\n", sc_info.round_durations)
    print("\nNumber of active tags:\n", sc_info.num_tags_active)
    print("\nID probs:\n", sc_info.id_probs)
