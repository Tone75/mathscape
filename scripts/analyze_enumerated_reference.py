"""RQ-M11 primary saved-data analysis, called only by the preregistered guard.

There is deliberately no standalone CLI. analyze validates ALL60 M10 originals
and panels before the first enumeration. Tests use synthetic fixed fixtures.
"""
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import stats

from mathscape import enumerated_reference as engine
from mathscape import matched_moments
from mathscape.small_sample import transform_raw

DISTRIBUTIONS = ('gamma', 'lognormal')
ACCEPTED_HEAD = 'bd01ae7ff55b1e8593a4b3bd003587db2732c050'
SOURCE_METHODS = ('t', 'percentile', 'bca')
SOURCE_NAMESPACE = 0x4D3130
N, R, K, B = 5, 20, 120, 4999
METHOD_INDICES = (2, 1)
PANEL_FLOAT_FIELDS = ('lower', 'upper', 'z0', 'a', 'q_low', 'q_high', 'tau_low', 'tau_high')
REFERENCE_ARRAY_FIELDS = ('q', 'E_cdf', 'E_linear', 'atom_index', 'atom_frequency',
    'cdf_left_count', 'cdf_right_count', 'cdf_left', 'cdf_right',
    'distance_left', 'distance_right', 'q_num', 'q_den',
    'distance_left_num', 'distance_left_den', 'distance_right_num', 'distance_right_den',
    'tail_mass', 'enum_tail_count', 'tail_warning')


def sample_hash(values):
    return sha256(np.asarray(values, dtype='<f8').tobytes(order='C')).hexdigest()


def _scalar(panel, name):
    value = np.asarray(panel[name])
    if value.shape != ():
        raise ValueError(f'non-scalar metadata: {name}')
    return value.item()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _load(path):
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def validate_inputs(source):
    """Validate identities/bytes and every panel before any enumerate call.

    Mathematical invalidity and unobserved slots remain method-local data.
    Shape, identity, raw transform or valid-diagnostic contradictions stop.
    Accepted Git blob / manifest/source/environment binding belongs to Root's
    outer guard and input_receipt, and is not duplicated in this module.
    """
    source = Path(source)
    receipt_path = source / 'originals/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt_hash = sha256(receipt_path.read_bytes()).hexdigest()
    _require(receipt['question_id'] == 'RQ-M10' and receipt['original_samples'] == 60,
             'M10 original receipt identity differs')
    expected_files = {f'{d}_seed{s}.npz' for d in DISTRIBUTIONS for s in (0, 1, 2)}
    _require(set(receipt['original_files_sha256']) == expected_files, 'original filenames differ')
    records = []
    hashes = set()
    receipt_records = {(row['distribution'], row['master_seed'], row['outer_id']): row
                       for row in receipt['records']}
    _require(len(receipt_records) == len(receipt['records']) == 60, 'original receipt duplicate/missing IDs')
    for distribution in DISTRIBUTIONS:
        for seed in (0, 1, 2):
            filename = f'{distribution}_seed{seed}.npz'
            original_path = source / 'originals' / filename
            _require(sha256(original_path.read_bytes()).hexdigest() == receipt['original_files_sha256'][filename],
                     'original NPZ hash differs from receipt')
            original = _load(original_path)
            panel = _load(source / 'panels' / filename)
            for name, expected in dict(distribution=distribution, master_seed=seed, n=N,
                                       namespace=SOURCE_NAMESPACE).items():
                _require(_scalar(original, name) == expected, f'original metadata differs: {name}')
                _require(_scalar(panel, name) == expected, f'panel metadata differs: {name}')
            for name, expected in dict(O=10, R=R, streams_per_budget=K, B=B, alpha=.05,
                                       cohort='M10').items():
                _require(_scalar(panel, name) == expected, f'panel design differs: {name}')
            _require(_scalar(panel, 'original_receipt_sha256') == receipt_hash, 'panel receipt binding differs')
            _require(tuple(panel['methods'].astype(str)) == SOURCE_METHODS, 'method axis differs')
            for name, expected, dtype in (
                    ('outer_id', np.arange(10), np.uint16),
                    ('budget_repeat_id', np.arange(R), np.uint8),
                    ('stream_id', np.arange(K), np.uint8)):
                _require(panel[name].dtype == dtype and np.array_equal(panel[name], expected), f'panel ID differs: {name}')
            _require(original['outer_id'].dtype == np.uint16 and np.array_equal(original['outer_id'], np.arange(10)),
                     'original ID differs')
            _require(original['original_keys'].dtype == np.uint32 and original['original_keys'].shape == (10, 6),
                     'original key shape/dtype differs')
            constants = matched_moments.population_info(distribution)['constants']
            _require(original['constant_names'].astype(str).tolist() == list(constants), 'analytic constant names differ')
            _require(original['constant_values'].dtype == np.dtype('<f8') and
                     original['constant_values'].tobytes() == np.asarray(list(constants.values()), dtype='<f8').tobytes(),
                     'analytic constant bits differ')
            for name in ('samples', 'raw'):
                _require(original[name].shape == (10, N) and original[name].dtype == np.dtype('<f8') and
                         np.isfinite(original[name]).all(), 'original sample/raw shape/dtype/nonfinite')
            _require(panel['samples'].dtype == np.dtype('<f8') and panel['samples'].shape == (10, N) and
                     panel['samples'].tobytes() == original['samples'].tobytes(), 'panel sample bytes differ')
            for field in PANEL_FLOAT_FIELDS:
                _require(panel[field].shape == (10, R, 3, K) and panel[field].dtype == np.float64,
                         f'panel float shape/dtype differs: {field}')
            for field in ('valid', 'tail_warning'):
                _require(panel[field].shape == (10, R, 3, K) and panel[field].dtype == bool,
                         f'panel mask differs: {field}')
            observed = panel['observed_streams']
            _require(observed.shape == (10, R, K) and observed.dtype == bool, 'observed mask differs')
            _require(panel['reason_codes'].dtype == np.uint8 and panel['reason_codes'].shape == (10, R, 3, K) and
                     panel['reason_labels'].dtype.kind == 'U' and np.all(panel['reason_codes'] < len(panel['reason_labels'])),
                     'reason code dictionary differs')
            reasons = panel['reason_labels'][panel['reason_codes']]
            for field in ('less_count', 'equal_count'):
                _require(panel[field].shape == (10, R, 3, K) and panel[field].dtype == np.uint16,
                         'tie count shape/dtype differs')
            for oi in range(10):
                sample = original['samples'][oi]
                identity = (distribution, seed, oi)
                key = (SOURCE_NAMESPACE, 1 if distribution == 'gamma' else 2, N, oi, 0, 0)
                _require(tuple(original['original_keys'][oi]) == key, 'original key differs')
                _require(transform_raw(original['raw'][oi], distribution).tobytes() == sample.tobytes(),
                         'saved raw transformation differs')
                digest = sample_hash(sample)
                _require(digest not in hashes, 'duplicate M10 original')
                hashes.add(digest)
                _require(original['sample_sha256'][oi].decode() == panel['sample_sha256'][oi].decode() == digest,
                         'sample hash differs')
                record = receipt_records[identity]
                _require(record['sample_sha256'] == digest and record['raw_sha256'] == sample_hash(original['raw'][oi]) and
                         tuple(record['spawn_key']) == key and record['n'] == N, 'receipt record differs')
                context = engine.sample_context(sample)
                for method_index in range(3):
                    mask = observed[oi] & panel['valid'][oi, :, method_index]
                    _require(np.all(reasons[oi, :, method_index][mask] == ''), 'valid slot has invalid reason')
                    invalid = observed[oi] & ~panel['valid'][oi, :, method_index]
                    _require(np.all(reasons[oi, :, method_index][invalid] != ''), 'invalid slot missing reason')
                    lower = panel['lower'][oi, :, method_index]
                    upper = panel['upper'][oi, :, method_index]
                    _require(np.isfinite(lower[mask]).all() and np.isfinite(upper[mask]).all() and
                             np.all(lower[mask] <= upper[mask]), 'valid endpoint nonfinite/reversed')
                    finite_a = panel['a'][oi, :, method_index][observed[oi]]
                    finite_a = finite_a[np.isfinite(finite_a)]
                    if np.isfinite(context['a']):
                        _require(np.allclose(finite_a, context['a'], rtol=engine.NUMERIC_RTOL, atol=engine.NUMERIC_ATOL),
                                 'saved acceleration differs from original')
                    if method_index == 0:
                        if context['valid']:
                            half = stats.t.ppf(.975, N - 1) * context['se']
                            _require(np.allclose(lower[mask], context['theta'] - half, rtol=engine.NUMERIC_RTOL, atol=engine.NUMERIC_ATOL) and
                                     np.allclose(upper[mask], context['theta'] + half, rtol=engine.NUMERIC_RTOL, atol=engine.NUMERIC_ATOL),
                                     'saved t interval differs from df4 reference')
                        continue
                    qlo = panel['q_low'][oi, :, method_index]
                    qhi = panel['q_high'][oi, :, method_index]
                    _require(np.isfinite(qlo[mask]).all() and np.isfinite(qhi[mask]).all() and
                             np.all((0 <= qlo[mask]) & (qlo[mask] < qhi[mask]) & (qhi[mask] <= 1)),
                             'valid saved q nonfinite/outside order')
                    tau_low = B * np.minimum(qlo, 1 - qlo)
                    tau_high = B * np.minimum(qhi, 1 - qhi)
                    _require(np.allclose(panel['tau_low'][oi, :, method_index][mask], tau_low[mask], rtol=engine.NUMERIC_RTOL, atol=engine.NUMERIC_ATOL) and
                             np.allclose(panel['tau_high'][oi, :, method_index][mask], tau_high[mask], rtol=engine.NUMERIC_RTOL, atol=engine.NUMERIC_ATOL),
                             'saved finiteB tau differs')
                    _require(np.array_equal(panel['tail_warning'][oi, :, method_index][mask],
                                            ((tau_low < 10) | (tau_high < 10))[mask]), 'saved warning differs')
                records.append(dict(distribution=distribution, master_seed=seed, outer_id=oi,
                    sample_sha256=digest, sample=sample.copy(), context=context,
                    panel={field: panel[field][oi] for field in (*PANEL_FLOAT_FIELDS, 'valid', 'tail_warning', 'less_count', 'equal_count')},
                    observed=observed[oi].copy(), reason=reasons[oi]))
    _require(len(records) == 60, 'all60 inputs required before enumeration')
    return records


def saved_arrays(record):
    panel = record['panel']
    result = dict(observed=record['observed'].reshape(-1),
                  budget_repeat_id=np.repeat(np.arange(R, dtype=np.uint8), K),
                  stream_id=np.tile(np.arange(K, dtype=np.uint8), R),
                  saved_t_lower=panel['lower'][:, 0, :].reshape(-1),
                  saved_t_upper=panel['upper'][:, 0, :].reshape(-1),
                  saved_t_valid=panel['valid'][:, 0, :].reshape(-1))
    result['saved_endpoint'] = np.stack([np.stack([panel[e][:, mi, :].reshape(-1) for e in engine.ENDPOINTS], axis=1)
                                         for mi in METHOD_INDICES], axis=1)
    result['saved_q'] = np.stack([np.stack([panel[q][:, mi, :].reshape(-1) for q in ('q_low', 'q_high')], axis=1)
                                  for mi in METHOD_INDICES], axis=1)
    for output_name, field in (('saved_valid', 'valid'), ('saved_warning', 'tail_warning'),
                               ('saved_z0', 'z0'), ('saved_a', 'a'), ('less_count', 'less_count'),
                               ('equal_count', 'equal_count')):
        result[output_name] = np.stack([panel[field][:, mi, :].reshape(-1) for mi in METHOD_INDICES], axis=1)
    result['saved_reason'] = np.stack([record['reason'][:, mi, :].reshape(-1) for mi in METHOD_INDICES], axis=1)
    for output_name, field in (('saved_tau_low', 'tau_low'), ('saved_tau_high', 'tau_high')):
        result[output_name] = np.stack([panel[field][:, mi, :].reshape(-1) for mi in METHOD_INDICES], axis=1)
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(_json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def _write_csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: 'NA' if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
                             for key, value in row.items()})


def _save_npz(path, payload):
    arrays = {key: np.asarray(value) for key, value in payload.items()}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError('object arrays forbidden')
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def summarize(original_rows, endpoint_rows):
    output = []
    for distribution in DISTRIBUTIONS:
        for method in engine.METHODS:
            group = [row for row in original_rows if row['distribution'] == distribution and row['method'] == method]
            row = dict(distribution=distribution, method=method, **engine.group_decision(group))
            complete = [item for item in group if item['complete']]
            for prefix, selected in (('', group if len(complete) == len(group) == 30 else []), ('conditional_', complete)):
                values = np.array([item['e'] for item in selected])
                for name, value in dict(e_mean=values.mean() if len(values) else np.nan,
                    e_p50=np.quantile(values, .5, method='linear') if len(values) else np.nan,
                    e_p95=np.quantile(values, .95, method='linear') if len(values) else np.nan,
                    e_max=values.max() if len(values) else np.nan).items():
                    row[prefix + name] = float(value)
            row['primary_hypothesis'] = method == 'bca'
            if method != 'bca':
                row['status'] = 'descriptive_only'
            output.append(row)
    return output


def rank_and_represent(original_rows):
    representatives = []
    for distribution in DISTRIBUTIONS:
        for method in engine.METHODS:
            group = [row for row in original_rows if row['distribution'] == distribution and row['method'] == method]
            complete = [row for row in group if row['complete']]
            for rank, row in enumerate(sorted(complete, key=lambda item: (item['e'], item['master_seed'], item['outer_id'])), 1):
                row['rank_e'] = rank
            if method == 'bca' and complete:
                for kind, key in (('minimum', lambda item: (item['e'], item['master_seed'], item['outer_id'])),
                                  ('maximum', lambda item: (-item['e'], item['master_seed'], item['outer_id']))):
                    selected = min(complete, key=key)
                    representatives.append(dict(selection=kind, **{field: selected[field] for field in
                        ('distribution', 'method', 'master_seed', 'outer_id', 'sample_sha256', 'e', 'margin', 'rank_e')}))
    return representatives


def analyze_records(records, output):
    """Writer shared with synthetic fixtures; public analyze always validates60."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ('references', 'comparisons'):
        (output / directory).mkdir(exist_ok=False)
    endpoint_rows, original_rows, reference_rows = [], [], []
    started = perf_counter()
    completed = []
    persisted = False
    try:
        for record in records:
            identity = {name: record[name] for name in ('distribution', 'master_seed', 'outer_id', 'sample_sha256')}
            stem = f"{record['distribution']}_seed{record['master_seed']}_outer{record['outer_id']}"
            reference = engine.build_reference(record['sample'])
            comparison = engine.stream_comparison(reference, saved_arrays(record))
            metadata = dict(**identity, methods=np.array(engine.METHODS), endpoints=np.array(engine.ENDPOINTS),
                            source_question_id='RQ-M10', question_id='RQ-M11', n=N, B=B, R=R, K=K)
            _save_npz(output / 'references' / (stem + '.npz'), {**reference, **metadata})
            _save_npz(output / 'comparisons' / (stem + '.npz'), {**comparison, **metadata})
            for mi, method in enumerate(engine.METHODS):
                local_endpoints = []
                for ei, endpoint in enumerate(engine.ENDPOINTS):
                    ref = dict(**identity, method=method, endpoint=endpoint, theta=reference['theta'],
                        se=reference['se'], a=reference['a'], enum_less_count=reference['enum_less_count'],
                        enum_equal_count=reference['enum_equal_count'], p_star=reference['p_star'],
                        z0_star=reference['z0_star'] if method == 'bca' else np.nan,
                        support_count=len(reference['support']), reference_valid=bool(reference['valid'][mi]),
                        reference_reason=str(reference['reason'][mi]))
                    ref.update({field: np.asarray(reference[field])[mi, ei].item() for field in REFERENCE_ARRAY_FIELDS})
                    reference_rows.append(ref)
                    measured = engine.endpoint_metrics(comparison, mi, ei)
                    valid_q = comparison['saved_q'][:, mi, ei]
                    valid_q = valid_q[np.isfinite(valid_q) & comparison['observed']]
                    measured.update(saved_q_min=float(valid_q.min()) if len(valid_q) else np.nan,
                                    saved_q_max=float(valid_q.max()) if len(valid_q) else np.nan)
                    endpoint_rows.append({**identity, 'method': method, 'endpoint': endpoint, **measured})
                    local_endpoints.append(measured)
                original_rows.append(dict(**identity, method=method, **engine.classify_original(local_endpoints),
                    rank_e=None, saved_warning_series=int((record['observed'].reshape(-1) & comparison['saved_warning'][:, mi]).sum()),
                    reference_valid=bool(reference['valid'][mi]), reference_reason=str(reference['reason'][mi])))
            completed.append(identity)
        representatives = rank_and_represent(original_rows)
        summaries = summarize(original_rows, endpoint_rows)
        groups = {row['distribution']: {key: value for key, value in row.items() if key not in ('distribution', 'method')}
                  for row in summaries if row['method'] == 'bca'}
        hypotheses = dict(question_id='RQ-M11', hypothesis='H11-E', target_method='bca',
            RMSE_limit=.10, required_fraction=.90, required_count=27, planned_N_per_distribution=30,
            original_n=5, series_per_original=2400, reference='conditional_binary64_ordered_enumeration_inverseCDF',
            computed_status=engine.joint_status([groups[d]['status'] for d in DISTRIBUTIONS]),
            status=engine.joint_status([groups[d]['status'] for d in DISTRIBUTIONS]), groups=groups,
            verification_required=True,
            interpretation='Fixed-data descriptive classification. Public conclusion requires completed run and passed independent audit; otherwise hold. No population90% guarantee or new t interval/pvalue.')
        _write_csv(output / 'reference_metrics.csv', reference_rows)
        _write_csv(output / 'endpoint_metrics.csv', endpoint_rows)
        _write_csv(output / 'original_metrics.csv', original_rows)
        _write_csv(output / 'summary.csv', summaries)
        _write_json(output / 'representatives.json', dict(rule='BCa perpopulation min/max e; ties seed,outer ascending', examples=representatives))
        _write_json(output / 'hypotheses.json', hypotheses)
        persisted = True
        return hypotheses
    finally:
        _write_json(output / 'analysis_resources.json', dict(question_id='RQ-M11', completed_originals=len(completed),
            completed_ids=completed, analysis_loop_seconds=perf_counter() - started,
            new_originals=0, new_random_bootstrap_streams=0, PRNG_replays=0,
            enumerated_ordered_rows=3125 * len(completed), enumerated_indices=15625 * len(completed),
            saved_common_series=2400 * len(completed), status='completed' if persisted else 'stopped',
            scope='Internal deterministic enumeration/comparison/output loop only; external guard owns full180second/RSS/50MiB measurements.'))


def analyze(source: Path, output: Path):
    """The only production callable; guard validates prereg before calling."""
    records = validate_inputs(source)
    return analyze_records(records, output)
