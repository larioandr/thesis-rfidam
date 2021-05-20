import multiprocessing
from time import perf_counter

import click
import pandas as pd
import numpy as np

from rfidam.chains import estimate_rounds_props, estimate_id_probs
from rfidam.protocol.symbols import TagEncoding, DR
from rfidam.scenario import parse_scenario, mark_scenario
from rfidam.simulation import ModelParams, build_scenario_info
from rfidam.protocol.protocol import Protocol, LinkProps
from rfidam.cy_ext.simulation import simulate as call_simulate


@click.group()
def cli():
    pass


@cli.group()
def batch():
    pass


_batch_options = [
    click.option('--tari', default=12.5, help="Tari value (default: 12.5)"),
    click.option('--rtcal', default=37.5, help="RTcal value (default: 37.5)"),
    click.option('--trcal', default=56.25, help="TRcal value (default: 56.25"),
    click.option('--tag-encoding', '-m', default=2,
                 help="M value, i.e. number of symbols per bit (default: 2)"),
    click.option('--dr', default='64/3', type=click.Choice(['8', '64/3']),
                 help="Division ratio (default: 64/3)"),
    click.option('-q', default=2, help="Q parameter value (default: 2)"),
    click.option('--time-in-area', default=2.42,
                 help="How long tag is in area (default: 0.1"),
    click.option('--time-off', default=0.1,
                 help="Power-off duration (default: 0.1)"),
    click.option('-j', '--num-workers', default=1,
                 help='Number of workers (default: 1)'),
    click.option('--jupyter', is_flag=True, default=False),
]


def add_options(options):
    # This command was taken from answer on StackOverflow:
    # https://stackoverflow.com/a/40195800/4563846
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


@batch.command()
@add_options(_batch_options)
@click.option('-n', '--num-tags', default=1000,
              help='Number of tags to simulate (default: 1000)')
@click.argument('file_name')
def simulate(file_name, **kwargs):
    _run_batch_command(file_name, kwargs, _apply_simulate, field_suffix='sim')


@batch.command()
@add_options(_batch_options)
@click.option('--ext-mul', default=100,
              help="Scenario length multiplier (default: 100)")
@click.argument('file_name')
def solve(file_name, **kwargs):
    _run_batch_command(file_name, kwargs, _apply_solve, field_suffix='ana')


def _run_batch_command(file_name, kwargs, fn, field_suffix):
    df = pd.read_csv(file_name)

    if kwargs['jupyter']:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # Split into chunks. Each chunk is computed in parallel, then chunks
    # are joined.
    n_workers = kwargs['num_workers']
    source_chunks = np.array_split(df, len(df) // n_workers)
    result_chunks = []

    for chunk in tqdm(source_chunks):
        sub_chunks = np.array_split(chunk, n_workers)
        sub_chunks = [c for c in sub_chunks if len(c) > 0]
        with multiprocessing.Pool(min(len(sub_chunks), n_workers)) as pool:
            results = pool.map(
                fn, [(sub_chunk, kwargs) for sub_chunk in sub_chunks])
        for sc, res in zip(sub_chunks, results):
            sc['__ret'] = res
        result_chunks.append(pd.concat(sub_chunks, ignore_index=True))

    df = pd.concat(result_chunks, ignore_index=True)
    for field in [f'p_{field_suffix}', f't_{field_suffix}']:
        df[field] = df.apply(lambda row: row['__ret'][field], axis=1)
    df.drop(['__ret'], axis=1).to_csv(file_name, index=False)


def _apply_simulate(args):
    """
    Run RFID simulation with parameters given in DataFrame.
    This function is intended to be executed by multiprocessing workers.

    Parameters
    ----------
    args : tuple (DataFrame, kwargs)
    """
    df, kwargs = args
    n_tags = kwargs['num_tags']

    def fn(data):
        use_tid = data['id_type'] == "TID"
        proto = _build_protocol(kwargs, use_tid)
        scenario = parse_scenario(data['scenario'])
        params = ModelParams(
            protocol=proto,
            arrival_interval=data['interval'],
            time_in_area=kwargs['time_in_area'],
            scenario=scenario,
            ber=data['ber'])

        t_start = perf_counter()
        journal = call_simulate(
            params, only_id=True, verbose=False, n_tags=n_tags)

        return {
            'p_sim': journal.p_id,
            't_sim': perf_counter() - t_start
        }

    if isinstance(df, pd.DataFrame):
        return df.apply(fn, axis=1)

    return fn(df)


def _apply_solve(args):
    """
    Run analytic ID probability estimation for a given DataFrame.
    This function is intended to be executed by multiprocessing workers.

    Parameters
    ----------
    args : tuple (DataFrame, kwargs)
    """
    df, kwargs = args
    ext_mul = kwargs['ext_mul']

    def fn(data):
        use_tid = data['id_type'] == "TID"
        proto = _build_protocol(kwargs, use_tid)
        scenario = parse_scenario(data['scenario']) * ext_mul
        ber = data['ber']
        arrival_interval = data['interval']
        time_in_area = kwargs['time_in_area']

        t_start = perf_counter()

        rounds_props = estimate_rounds_props(
            scenario,
            protocol=proto,
            ber=ber,
            arrival_interval=arrival_interval,
            time_in_area=time_in_area,
            t0=(100.5 * arrival_interval))

        marked_scenario = mark_scenario(
            scenario,
            arrival_interval=arrival_interval,
            time_in_area=time_in_area,
            durations=rounds_props['durations'],
            t0=(100.5 * arrival_interval))

        p_id, _ = estimate_id_probs(
            rounds_props['n_active_tags'],
            marked_scenario,
            durations=rounds_props['durations'],
            ber=ber,
            protocol=proto,
            time_in_area=time_in_area)

        return {
            'p_ana': p_id,
            't_ana': perf_counter() - t_start
        }

    if isinstance(df, pd.DataFrame):
        return df.apply(fn, axis=1)

    return fn(df)


def _build_protocol(kwargs, use_tid):
    props = LinkProps(
        tari=kwargs['tari'],
        rtcal=kwargs['rtcal'],
        trcal=kwargs['trcal'],
        m=TagEncoding.parse(kwargs['tag_encoding']),
        dr=DR.parse(kwargs['dr']),
        trext=kwargs.get('trext', False),
        q=kwargs['q'],
        use_tid=use_tid,
        t_off=kwargs['time_off'],
        n_data_words=kwargs.get('n_data_words', 4),
        n_epcid_bytes=kwargs.get('n_epcid_bytes', 12))
    return Protocol(props)


def main():
    cli()
    # props = LinkProps(
    #     tari=12.5e-6,
    #     rtcal=25e-6,
    #     trcal=40e-6,
    #     m=TagEncoding.M2,
    #     dr=DR.DR_643,
    #     trext=False,
    #     q=2,
    #     use_tid=False)
    # protocol = Protocol(props)
    # params = ModelParams(
    #     protocol,
    #     arrival_interval=1.0,
    #     time_in_area=2.3,
    #     scenario=parse_scenario("ABABx"),
    #     ber=.02)
    #
    # journal = simulate(params, n_tags=1000)
    # sc_info = build_scenario_info([journal], 4)
    # print("P_ID =", journal.p_id, f"({journal.n_identified}/{journal.n_tags})")
    # print("\nRounds durations:\n", sc_info.round_durations)
    # print("\nNumber of active tags:\n", sc_info.num_tags_active)
    # print("\nID probs:\n", sc_info.id_probs)


if __name__ == '__main__':
    cli()
