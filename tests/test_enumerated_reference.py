"""Deterministic synthetic fixtures only; no accepted M10 numerical analysis."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from itertools import product
from math import factorial
from pathlib import Path
import json

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from mathscape import enumerated_reference as e
from mathscape import matched_moments, small_sample, stability
from scripts import analyze_enumerated_reference as runner


@pytest.fixture(autouse=True)
def forbid_real_random(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('M11 tests forbid RNG/main-original generation')
    for name in ('SeedSequence', 'PCG64', 'Generator', 'default_rng'):
        monkeypatch.setattr(np.random, name, forbidden)
    monkeypatch.setattr(small_sample, 'generate_cohort', forbidden)
    monkeypatch.setattr(small_sample, 'draw_original', forbidden)


def fixed_saved(sample, repetitions=2400):
    reference = e.build_reference(sample)
    # Deliberately prescribed endpoint errors, never draws or a real M10 panel.
    wave = np.linspace(-.06, .06, repetitions)
    endpoint = np.broadcast_to(reference['E_cdf'], (repetitions, 2, 2)).copy()
    endpoint[:, :, 0] += reference['se'] * wave[:, None]
    endpoint[:, :, 1] += reference['se'] * (wave[:, None] + .01)
    saved = dict(saved_endpoint=endpoint, saved_valid=np.ones((repetitions, 2), bool),
        observed=np.ones(repetitions, bool), saved_q=np.broadcast_to(reference['q'], endpoint.shape).copy(),
        saved_z0=np.column_stack([np.full(repetitions, reference['z0_star']), np.full(repetitions, np.nan)]),
        saved_a=np.full((repetitions, 2), reference['a']))
    return reference, saved


def fake_record(seed=0, outer=0):
    sample = np.array([-.9, -.4, .1, .6, 1.8]) + .01 * (10 * seed + outer)
    context = stability.prepare_sample(sample)
    means = np.resize(np.array([-.4, -.1, context['mean'], .4, .8]), 4999)
    inferred = small_sample.infer_single(context, means)
    panel = {field: np.empty((20, 3, 120), dtype=float) for field in runner.PANEL_FLOAT_FIELDS}
    panel.update(valid=np.empty((20, 3, 120), bool), tail_warning=np.empty((20, 3, 120), bool),
                 less_count=np.empty((20, 3, 120), np.uint16), equal_count=np.empty((20, 3, 120), np.uint16))
    reasons = np.full((20, 3, 120), '', dtype='U96')
    for mi, method in enumerate(runner.SOURCE_METHODS):
        for field in panel:
            panel[field][:, mi, :] = inferred[method][field]
        reasons[:, mi, :] = inferred[method]['reason']
    return dict(distribution='gamma', master_seed=seed, outer_id=outer, sample=sample,
                sample_sha256=runner.sample_hash(sample), context=e.sample_context(sample),
                panel=panel, reason=reasons, observed=np.ones((20, 120), bool))


def test_all3125_ordered_positions_and126_multinomial_weights():
    indices = e.ordered_indices()
    expected = np.asarray(list(product(range(5), repeat=5)), dtype=np.uint8)
    np.testing.assert_array_equal(indices, expected)
    counts = Counter(tuple(np.bincount(row, minlength=5)) for row in indices)
    assert len(counts) == 126 and sum(counts.values()) == 3125
    for vector, frequency in counts.items():
        denominator = np.prod([factorial(int(count)) for count in vector])
        assert frequency == factorial(5) // denominator


def test_permutation_rounding_prevents_representative_mean_times_weight():
    sample = np.array([1e16, 1., -1e16, 3., -7.])
    dist = e.enumerate_distribution(sample)
    first, second = np.array([0, 1, 2, 3, 4]), np.array([0, 2, 1, 3, 4])
    assert sample[first].mean() != sample[second].mean()
    rows = np.all(np.sort(dist['indices'], axis=1) == np.arange(5), axis=1)
    assert rows.sum() == 120 and len(np.unique(dist['means'][rows])) > 1
    assert dist['frequency'].sum() == 3125
    np.testing.assert_array_equal(dist['means'], np.mean(sample[dist['indices']], axis=1))


def test_repeated_values_keep_position_multiplicity_and_signed_zero_one_atom():
    sample = np.array([-0., 0., 0., 1., 1.])
    dist = e.enumerate_distribution(sample)
    assert dist['indices'].shape == (3125, 5)
    assert dist['frequency'].sum() == 3125
    zero = np.flatnonzero(dist['support'] == 0)
    assert len(zero) == 1 and dist['frequency'][zero[0]] == 3**5
    assert not np.signbit(dist['support'][zero[0]])


def test_exact_rational_cdf_boundary_and_both_nextafter_neighbors():
    support, frequency = np.array([0., 1.]), np.array([1, 4])
    q = .2  # binary64 .2 is strictly greater than mathematical1/5.
    assert Fraction.from_float(q) > Fraction(1, 5)
    assert e.inverse_cdf(support, frequency, q)['atom_index'] == 1
    assert e.inverse_cdf(support, frequency, np.nextafter(q, 0))['atom_index'] == 0
    assert e.inverse_cdf(support, frequency, np.nextafter(q, 1))['atom_index'] == 1
    selected = e.inverse_cdf(support, frequency, q)
    distance = Fraction(int(selected['distance_left_num']), int(selected['distance_left_den']))
    assert distance == Fraction.from_float(q) - Fraction(1, 5)
    assert e.inverse_cdf(support, frequency, 0)['value'] == 0
    assert e.inverse_cdf(support, frequency, 1)['value'] == 1
    exact = e.inverse_cdf(support, np.array([1, 1]), .5)
    assert exact['atom_index'] == 0 and exact['distance_right_num'] == '0'


@pytest.mark.parametrize('q', [-.1, 1.1, np.inf, np.nan, True])
def test_inverse_rejects_invalid_probability_without_clipping(q):
    with pytest.raises(ValueError):
        e.inverse_cdf(np.array([0., 1.]), np.array([1, 1]), q)


def test_full_reference_half_tie_a_linear_uses_same3125_not_finiteB_z0():
    sample = np.array([-1., -.6, .1, .7, 2.2])
    ref = e.build_reference(sample)
    context = stability.prepare_sample(sample)
    accepted = stability.infer_prefixes(context, ref['means'], (3125,))[3125]
    probability = (np.count_nonzero(ref['means'] < sample.mean()) + .5 * np.count_nonzero(ref['means'] == sample.mean())) / 3125
    assert ref['p_star'] == probability and ref['z0_star'] == ndtri(probability)
    assert ref['a'] == context['a']
    for mi, method in enumerate(e.METHODS):
        assert ref['valid'][mi] == accepted[method]['valid']
        np.testing.assert_array_equal(ref['q'][mi], [accepted[method]['q_low'], accepted[method]['q_high']])
        np.testing.assert_array_equal(ref['E_linear'][mi], [accepted[method]['lower'], accepted[method]['upper']])
        for ei in (0, 1):
            chosen = e.inverse_cdf(ref['support'], ref['frequency'], ref['q'][mi, ei])
            assert ref['E_cdf'][mi, ei] == chosen['value']
            assert ref['cdf_left_count'][mi, ei] == chosen['cdf_left_count']


def test_constant_sample_has_percentile_reference_but_invalid_SE_and_BCa():
    ref = e.build_reference(np.ones(5))
    assert ref['frequency'].tolist() == [3125]
    assert not ref['valid'][0] and ref['valid'][1]
    np.testing.assert_array_equal(ref['E_cdf'][1], [1., 1.])
    assert not np.isfinite(ref['se'])
    with pytest.raises(ValueError, match='nonfinite'):
        e.build_reference(np.array([0., 1., 2., 3., np.nan]))


def test_direct_rmse_ddof0_identity_and_component_cross_term():
    ref, saved = fixed_saved(np.array([-.9, -.4, .1, .6, 1.8]))
    comparison = e.stream_comparison(ref, saved)
    metric = e.endpoint_metrics(comparison, 0, 1)
    delta = comparison['delta'][:, 0, 1]
    assert metric['complete'] and metric['identity_ok']
    assert metric['b'] == pytest.approx(delta.mean())
    assert metric['rmse'] == pytest.approx(np.sqrt(np.mean(delta**2)))
    assert metric['variance_ddof0'] == pytest.approx(np.mean((delta - delta.mean())**2))
    assert metric['rmse']**2 == pytest.approx(metric['b']**2 + metric['variance_ddof0'])
    assert metric['mse'] == pytest.approx(metric['linear_mse'] + metric['convention_mse'] + metric['cross_term'])
    assert metric['rmse_identity_scale'] >= 0


def test_required_missing_and_method_local_invalid_preserve_planned2400():
    ref, saved = fixed_saved(np.array([-.9, -.4, .1, .6, 1.8]))
    saved['saved_valid'][137, 0] = False
    c = e.stream_comparison(ref, saved)
    bca = e.endpoint_metrics(c, 0, 0)
    pct = e.endpoint_metrics(c, 1, 0)
    assert not bca['complete'] and bca['valid_series'] == 2399 and bca['planned_series'] == 2400
    assert np.isnan(bca['rmse']) and np.isfinite(bca['conditional_rmse'])
    assert pct['complete']
    saved['observed'][19] = False
    c = e.stream_comparison(ref, saved)
    assert e.endpoint_metrics(c, 1, 1)['missing_series'] == 1


def test_metric_overflow_is_NA_not_complete():
    comparison = dict(observed=np.ones(2400, bool), error_valid=np.ones((2400, 2, 2), bool),
        delta=np.full((2400, 2, 2), 1e200), linear_delta=np.full((2400, 2, 2), 1e200),
        convention_offset=np.zeros((2400, 2, 2)))
    with np.errstate(over='ignore', invalid='ignore'):
        result = e.endpoint_metrics(comparison, 0, 0)
    assert not result['complete'] and result['reason'] == 'derived_metric_nonfinite'
    assert not np.isfinite(result['rmse'])


def test_identity_failure_is_code_error_not_a_new_scientific_gate(monkeypatch):
    ref, saved = fixed_saved(np.array([-.9, -.4, .1, .6, 1.8]))
    comparison = e.stream_comparison(ref, saved)
    original = e._moments
    def broken(*args):
        return original(*args) | dict(identity_ok=False)
    monkeypatch.setattr(e, '_moments', broken)
    with pytest.raises(ArithmeticError):
        e.endpoint_metrics(comparison, 0, 0)


def test_threshold_contact_exact_no_tolerance_and_fixed_group27of30():
    endpoints = [dict(complete=True, identity_ok=True, rmse=.10, missing_series=0, invalid_series=0)] * 2
    at = e.classify_original(endpoints)
    assert at['pass_flag'] is True and at['margin'] == 0
    above = e.classify_original([endpoints[0], endpoints[1] | dict(rmse=np.nextafter(.10, np.inf))])
    assert above['pass_flag'] is False
    originals = [dict(master_seed=i // 10, outer_id=i % 10, complete=True, pass_flag=i < 27, invalid=False, missing=False) for i in range(30)]
    assert e.group_decision(originals)['status'] == 'supported'
    originals[26]['pass_flag'] = False
    assert e.group_decision(originals)['status'] == 'not_supported'
    originals[29].update(complete=False, pass_flag=None, invalid=True)
    hold = e.group_decision(originals)
    assert hold['status'] == 'inconclusive' and hold['planned_N'] == 30 and hold['invalid_N'] == 1
    assert e.group_decision(originals[:-1])['missing_N'] == 1
    assert e.joint_status(['supported', 'inconclusive']) == 'inconclusive'
    assert e.joint_status(['not_supported', 'inconclusive']) == 'not_supported'


def test_saved_series_mapping_no_allocation_triple_count():
    record = fake_record()
    saved = runner.saved_arrays(record)
    assert saved['saved_endpoint'].shape == (2400, 2, 2)
    assert saved['budget_repeat_id'][120] == 1 and saved['stream_id'][120] == 0
    assert saved['saved_endpoint'][121, 0, 0] == record['panel']['lower'][1, 2, 1]
    assert saved['saved_endpoint'][121, 1, 1] == record['panel']['upper'][1, 1, 1]


def test_fixture_writer_schema_and_percentile_descriptive_only(tmp_path):
    result = runner.analyze_records([fake_record()], tmp_path)
    assert result['status'] == 'inconclusive' and result['groups']['gamma']['planned_N'] == 30
    with np.load(tmp_path / 'references/gamma_seed0_outer0.npz', allow_pickle=False) as saved:
        assert saved['means'].shape == (3125,) and saved['indices'].shape == (3125, 5)
        assert saved['q'].shape == (2, 2) and saved['valid'].shape == (2,)
    with np.load(tmp_path / 'comparisons/gamma_seed0_outer0.npz', allow_pickle=False) as saved:
        assert saved['delta'].shape == (2400, 2, 2)
    import csv
    summary = list(csv.DictReader((tmp_path / 'summary.csv').open()))
    assert len(summary) == 4
    assert all(row['status'] == 'descriptive_only' for row in summary if row['method'] == 'percentile')
    original = list(csv.DictReader((tmp_path / 'original_metrics.csv').open()))
    assert len(original) == 2
    resources = json.loads((tmp_path / 'analysis_resources.json').read_text())
    assert resources['status'] == 'completed' and resources['saved_common_series'] == 2400
    with pytest.raises(FileExistsError):
        runner.analyze_records([fake_record()], tmp_path)


def test_all60_validation_barrier_before_enumeration(monkeypatch, tmp_path):
    events = []
    def rejected(source):
        events.append('all60validation')
        raise ValueError('last original ID malformed')
    monkeypatch.setattr(runner, 'validate_inputs', rejected)
    monkeypatch.setattr(e, 'build_reference', lambda *args: pytest.fail('enumeration before all60validation'))
    with pytest.raises(ValueError):
        runner.analyze(tmp_path / 'source', tmp_path / 'out')
    assert events == ['all60validation'] and not (tmp_path / 'out').exists()


def test_failed_final_persistence_marks_stopped_not_completed(monkeypatch, tmp_path):
    original = runner._write_json
    def fail_hypothesis(path, payload):
        if Path(path).name == 'hypotheses.json':
            raise OSError('fixture final write failure')
        return original(path, payload)
    monkeypatch.setattr(runner, '_write_json', fail_hypothesis)
    with pytest.raises(OSError):
        runner.analyze_records([fake_record()], tmp_path)
    resources = json.loads((tmp_path / 'analysis_resources.json').read_text())
    assert resources['completed_originals'] == 1 and resources['status'] == 'stopped'


def test_representative_ties_seed_outer_ascending_not_reversed_max():
    rows = [dict(distribution='gamma', method='bca', master_seed=seed, outer_id=outer,
                 sample_sha256='fixed', complete=True, e=value, margin=.1-value, rank_e=None)
            for seed, outer, value in ((1, 0, .2), (0, 9, .2), (0, 0, .01))]
    selected = runner.rank_and_represent(rows)
    maximum = next(row for row in selected if row['selection'] == 'maximum')
    assert maximum['master_seed'] == 0 and maximum['outer_id'] == 9
