"""Independent RQ-M11 saved-data audit; no primary implementation or RNG.

Enumeration uses integer compositions and distinct position permutations, rather
than the primary base-five generator. Shared primitives are NumPy binary64 row
means/linear quantiles and SciPy ndtr/ndtri/Student t for accepted M10 checks.
"""
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy import special,stats

ROOT=Path(__file__).resolve().parents[1]
DISTRIBUTIONS=('gamma','lognormal')
METHODS=('t','percentile','bca')
REFERENCE_METHODS=('bca','percentile')
ENDPOINTS=('lower','upper')
O,R,STREAMS,N,B=10,20,120,5,4999
ENUMERATED=3125
UNKNOWN_COUNT=65535
MAIN_NAMESPACE=0x4D3130
ACCEPTED_INPUT='1bf925b55a838650ee020fe09f806cf64e8ebc02'
ACCEPTED_M10='bd01ae7ff55b1e8593a4b3bd003587db2732c050'
NUMERIC_RTOL,NUMERIC_ATOL=2e-10,2e-12
IDENTITY_RTOL,IDENTITY_ATOL=2e-12,2e-12

def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

def sample_hash(sample):
    return hashlib.sha256(np.asarray(sample, dtype='<f8').tobytes(order='C')).hexdigest()

def _original_se(samples):
    """Reconstruct early sample rejection before a rounded variance is used.

    Exactly constant fractional values can produce tiny positive np.std values.
    Such samples were rejected before inference, so their saved SE is undefined.
    """
    samples = np.asarray(samples, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        se = np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
        rejected = ~np.isfinite(samples).all(axis=1) | np.all(samples == samples[:, :1], axis=1) | ~np.isfinite(np.mean(samples, axis=1))
    se[rejected] = np.nan
    return se

def _meta(panel, name):
    value = np.asarray(panel[name])
    if value.ndim != 0:
        raise ValueError(f'metadata {name} must be scalar')
    return value.item()

def compare_rows(expected, observed, key_fields, *, label='table'):
    """Compare every independently calculable field, including NA and counts."""
    errors = []

    def key(row):
        return tuple((str(row[field]) for field in key_fields))
    expected_index = {key(row): row for row in expected}
    observed_index = {key(row): row for row in observed}
    if len(expected_index) != len(expected):
        errors.append(f'{label}: duplicate independently reconstructed keys')
    if len(observed_index) != len(observed):
        errors.append(f'{label}: duplicate saved keys')
    if expected_index.keys() != observed_index.keys():
        errors.append(f'{label}: missing/extra keys ({len(expected_index)} expected; {len(observed_index)} saved)')
    for row_key in expected_index.keys() & observed_index.keys():
        actual = observed_index[row_key]
        for field, reference in expected_index[row_key].items():
            if field not in actual:
                errors.append(f'{label}/{row_key}: missing column {field}')
                continue
            value = actual[field]
            if isinstance(reference, (bool, np.bool_)):
                matches = str(value).lower() in (('true', '1') if reference else ('false', '0'))
            elif reference is None or isinstance(reference, (int, float, np.integer, np.floating)):
                target, number = (finite_number(reference), finite_number(value))
                if target is None:
                    if isinstance(reference, (float, np.floating)) and np.isinf(reference):
                        try:
                            matches = bool(np.isinf(float(value)) and np.signbit(float(value)) == np.signbit(reference))
                        except (TypeError, ValueError):
                            matches = False
                    else:
                        try:
                            matches = bool(np.isnan(float(value)))
                        except (TypeError, ValueError):
                            matches = value is None or str(value) in ('', 'NA', 'None')
                elif number is None:
                    matches = False
                elif isinstance(reference, (int, np.integer)):
                    matches = number == target
                else:
                    matches = np.isclose(number, target, atol=2e-12, rtol=2e-10)
            else:
                matches = str(value) == str(reference)
            if not matches:
                errors.append(f'{label}/{row_key}: {field} differs')
    return errors

def compare_json(expected, observed, *, location='hypotheses'):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [f'{location}: expected object']
        for key, value in expected.items():
            if key not in observed:
                errors.append(f'{location}: missing {key}')
            else:
                errors.extend(compare_json(value, observed[key], location=f'{location}/{key}'))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(observed, list) or len(expected) != len(observed):
            return [f'{location}: list length/type differs']
        for index, (first, second) in enumerate(zip(expected, observed)):
            errors.extend(compare_json(first, second, location=f'{location}/{index}'))
    else:
        mismatch = compare_rows([dict(id=0, value=expected)], [dict(id=0, value=observed)], ['id'], label=location)
        if mismatch:
            errors.append(f'{location}: value differs')
    return errors

def _csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def _reason_array(panel):
    codes, labels = (panel['reason_codes'], panel['reason_labels'].astype(str))
    if codes.dtype.kind not in 'ui' or np.any(codes < 0) or np.any(codes >= labels.size):
        raise ValueError('reason code is outside the saved dictionary')
    return labels[codes]

def jackknife_acceleration(sample):
    """Delete-one-mean acceleration, with the common scale cancelled first."""
    values = np.asarray(sample, dtype=float)
    centered = values - values.mean()
    scale = np.max(np.abs(centered))
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        normalized = centered / scale
        return float(np.sum(normalized ** 3) / (6 * np.sum(normalized ** 2) ** 1.5))

def quantile_record(values, qlo, qhi, **details):
    """Linear empirical interval, allowing exact q endpoints without clipping."""
    values = np.asarray(values, dtype=float)
    base = dict(lower=np.nan, upper=np.nan, z0=np.nan, a=np.nan, q_low=np.nan, q_high=np.nan, tau_low=np.nan, tau_high=np.nan, valid=False, reason='', tail_warning=False, less_count=UNKNOWN_COUNT, equal_count=UNKNOWN_COUNT) | details
    if not np.isfinite([qlo, qhi]).all() or not 0 <= qlo < qhi <= 1:
        return base | dict(reason='bca_quantiles_invalid', q_low=float(qlo), q_high=float(qhi))
    with np.errstate(over='ignore', invalid='ignore'):
        ends = np.quantile(values, [qlo, qhi], method='linear')
    if not np.isfinite(ends).all() or ends[0] > ends[1]:
        return base | dict(reason='interval_endpoints_nonfinite', q_low=float(qlo), q_high=float(qhi))
    tau_lo = float(values.size * min(qlo, 1 - qlo))
    tau_hi = float(values.size * min(qhi, 1 - qhi))
    return base | dict(lower=float(ends[0]), upper=float(ends[1]), valid=True, q_low=float(qlo), q_high=float(qhi), tau_low=tau_lo, tau_high=tau_hi, tail_warning=min(tau_lo, tau_hi) < 10)

def reference_intervals(sample, means):
    """n5 t/linear-percentile/BCa from fixed arrays, including invalid reasons."""
    sample, means = (np.asarray(sample), np.asarray(means))
    empty = dict(lower=np.nan, upper=np.nan, z0=np.nan, a=np.nan, q_low=np.nan, q_high=np.nan, tau_low=np.nan, tau_high=np.nan, valid=False, reason='', tail_warning=False, less_count=UNKNOWN_COUNT, equal_count=UNKNOWN_COUNT)
    result = {method: empty.copy() for method in METHODS}
    reason = ''
    if sample.shape != (N,) or sample.dtype.kind not in 'iuf':
        reason = 'sample_not_one_dimensional_real'
    elif not np.isfinite(sample).all():
        reason = 'sample_nonfinite'
    elif np.all(sample == sample[0]):
        reason = 'sample_constant'
    if reason:
        return {method: record | dict(reason=reason) for method, record in result.items()}
    sample = sample.astype(float)
    mean = float(sample.mean())
    se = float(_original_se(sample[None, :])[0])
    if not np.isfinite(mean) or not np.isfinite(se) or se <= 0:
        return {method: record | dict(reason='sample_se_nonpositive_or_nonfinite') for method, record in result.items()}
    a = jackknife_acceleration(sample)
    critical = stats.t.ppf(0.975, N - 1)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        lower, upper = (mean - critical * se, mean + critical * se)
        tvalue = np.divide(mean, se)
    if np.isfinite([lower, upper, tvalue]).all():
        result['t'].update(lower=float(lower), upper=float(upper), valid=True, a=a)
    else:
        result['t'].update(a=a, reason='t_inference_undefined')
    for method in ('percentile', 'bca'):
        result[method]['a'] = a
    mean_reason = ''
    if means.ndim != 1 or means.dtype.kind not in 'iuf':
        mean_reason = 'bootstrap_distribution_not_one_dimensional_real'
    elif means.size < 1 or means.size > B:
        mean_reason = 'bootstrap_distribution_size_out_of_range'
    elif not np.isfinite(means).all():
        mean_reason = 'bootstrap_distribution_nonfinite'
    if mean_reason:
        for method in ('percentile', 'bca'):
            result[method]['reason'] = mean_reason
        return result
    values = means.astype(float)
    less = int(np.count_nonzero(values < mean))
    equal = int(np.count_nonzero(values == mean))
    diagnostics = dict(less_count=less, equal_count=equal)
    for method in METHODS:
        result[method].update(diagnostics)

    def record(qlo, qhi, **details):
        return quantile_record(values, qlo, qhi, a=a, **diagnostics, **details)
    result['percentile'] = record(0.025, 0.975)
    p = (less + 0.5 * equal) / values.size
    z0 = float(special.ndtri(p))
    result['bca'].update(z0=z0)
    if not np.isfinite(z0):
        result['bca']['reason'] = 'bca_bias_correction_nonfinite'
    elif not np.isfinite(a):
        result['bca']['reason'] = 'bca_acceleration_undefined'
    else:
        z_lower = special.ndtri(0.025)
        bounds = np.array([z_lower, -z_lower])
        denominator = 1 - a * (z0 + bounds)
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            result['bca']['reason'] = 'bca_adjustment_singular'
        else:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                q = special.ndtr(z0 + (z0 + bounds) / denominator)
            result['bca'] = record(q[0], q[1], z0=z0)
    return result

def inspect_panel(panel, strict_design=True):
    errors = []
    check = lambda condition, message: errors.append(message) if not condition else None
    samples = panel['samples']
    r, n = samples.shape
    j = panel['lower'].shape[1]
    k = panel['lower'].shape[3]
    shape = (r, j, 3, k)
    check(samples.dtype == np.dtype('float64') and panel['sample_sha256'].dtype == np.dtype('S64'), 'original sample/hash dtype differs')
    check(panel['outer_id'].dtype == np.dtype('uint16') and panel['budget_repeat_id'].dtype == np.dtype('uint8') and (panel['stream_id'].dtype == np.dtype('uint8')), 'ID dtype differs')
    check(panel['reason_codes'].dtype == np.dtype('uint8') and panel['reason_labels'].dtype.kind == 'U' and (len(panel['reason_labels']) <= 256), 'reason dictionary dtype/size differs')
    for field in ('lower', 'upper', 'a', 'z0', 'q_low', 'q_high', 'tau_low', 'tau_high'):
        check(panel[field].dtype == np.dtype('float64'), f'{field} dtype differs')
    if 'observed_streams' in panel:
        check(panel['observed_streams'].shape == (r, j, k) and panel['observed_streams'].dtype.kind == 'b', 'observed stream shape/dtype differs')
        check(not np.any(panel['valid'] & ~panel['observed_streams'][:, :, None, :]), 'unobserved interval marked valid')
    elif strict_design:
        errors.append('observed stream mask missing')
    check(n == N and _meta(panel, 'n') == N and (_meta(panel, 'alpha') == 0.05) and (_meta(panel, 'B') == 4999), 'fixed n/alpha/B differ')
    check(np.array_equal(panel['outer_id'], np.arange(r)), 'outer IDs differ from fixed prefix')
    check(np.array_equal(panel['budget_repeat_id'], np.arange(j)), 'budget repeat IDs differ')
    check(np.array_equal(panel['stream_id'], np.arange(k)), 'stream IDs differ')
    check(list(panel['methods'].astype(str)) == list(METHODS), 'method axis differs')
    if strict_design:
        check((r, j, k) == (10, 20, 120) and _meta(panel, 'O') == 10 and (_meta(panel, 'R') == 20) and (_meta(panel, 'streams_per_budget') == 120), 'main O/R/streams differ')
        check(_meta(panel, 'namespace') == MAIN_NAMESPACE and _meta(panel, 'distribution') in DISTRIBUTIONS and (_meta(panel, 'master_seed') in (0, 1, 2)), 'main namespace/population/seed differ')
        check(_meta(panel, 'reference_accepted_head') == ACCEPTED_INPUT and _meta(panel, 'cohort') == 'M10', 'reference head/cohort differs')
    for name in ('lower', 'upper', 'a', 'z0', 'q_low', 'q_high', 'tau_low', 'tau_high', 'valid', 'tail_warning', 'reason_codes'):
        check(panel[name].shape == shape, f'{name} shape differs from common original/replicate/method/inner grid')
    if errors:
        return errors
    valid, lo, hi = (panel['valid'], panel['lower'], panel['upper'])
    reason = _reason_array(panel)
    check(valid.dtype.kind == 'b' and panel['tail_warning'].dtype.kind == 'b', 'mask dtype differs')
    check(np.isfinite(lo[valid]).all() and np.isfinite(hi[valid]).all() and np.all(lo[valid] <= hi[valid]), 'valid endpoints nonfinite/reversed')
    check(np.isnan(lo[~valid]).all() and np.isnan(hi[~valid]).all(), 'invalid endpoints substituted')
    check(np.all(reason[valid] == '') and np.all(reason[~valid] != ''), 'reason/valid dictionary differs')
    hashes = panel['sample_sha256'].astype(str)
    check(hashes.shape == (r,) and all((hashes[i] == sample_hash(sample) for i, sample in enumerate(samples))), 'original hash mismatch')
    se = _original_se(samples)
    original_valid = np.isfinite(se) & (se > 0)
    critical = stats.t.ppf(0.975, N - 1)
    for oi, sample in enumerate(samples):
        observed = valid[oi, :, 0] & np.isfinite(lo[oi, :, 0]) & np.isfinite(hi[oi, :, 0])
        if not original_valid[oi]:
            check(not observed.any(), f'outer{oi}: invalid original has a valid t endpoint')
            continue
        mean = np.mean(sample)
        observed_lo, observed_hi = (lo[oi, :, 0][observed], hi[oi, :, 0][observed])
        check(np.allclose(observed_lo, mean - critical * se[oi], atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL) and np.allclose(observed_hi, mean + critical * se[oi], atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL), f'outer{oi}: t reference differs')
        check(not observed_lo.size or (np.all(observed_lo == observed_lo[0]) and np.all(observed_hi == observed_hi[0])), f'outer{oi}: observed t varies by replicate/inner')
        a = jackknife_acceleration(sample)
        stored = panel['a'][oi]
        finite = np.isfinite(stored)
        check(np.allclose(stored[finite], a, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL), f'outer{oi}: acceleration differs from independent jackknife')
        check(np.isfinite(panel['a'][oi, :, 2][valid[oi, :, 2]]).all(), f'outer{oi}: valid BCa acceleration undefined')
    for mi, method in enumerate(METHODS):
        mask = valid[:, :, mi]
        if method == 't':
            check(not panel['tail_warning'][:, :, mi].any(), 't has bootstrap tail warning')
            continue
        qlo, qhi = (panel['q_low'][:, :, mi], panel['q_high'][:, :, mi])
        check(np.isfinite(qlo[mask]).all() and np.isfinite(qhi[mask]).all() and np.all((qlo[mask] >= 0) & (qlo[mask] < qhi[mask]) & (qhi[mask] <= 1)), f'{method}: invalid quantiles')
        if method == 'percentile':
            check(np.allclose(qlo[mask], 0.025) and np.allclose(qhi[mask], 0.975), 'percentile nominal q differ')
        else:
            z0, a = (panel['z0'][:, :, mi][mask], panel['a'][:, :, mi][mask])
            check(np.isfinite(z0).all(), 'BCa valid z0 nonfinite')
            for probability, observed in ((0.025, qlo[mask]), (0.975, qhi[mask])):
                z = stats.norm.ppf(probability)
                den = 1 - a * (z0 + z)
                expected = special.ndtr(z0 + (z0 + z) / den)
                check(np.all(den > 0) and np.allclose(observed, expected, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL), 'BCa adjusted quantile formula differs')
        l_tau, u_tau = (4999 * np.minimum(qlo, 1 - qlo), 4999 * np.minimum(qhi, 1 - qhi))
        check(np.allclose(panel['tau_low'][:, :, mi][mask], l_tau[mask], atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL) and np.allclose(panel['tau_high'][:, :, mi][mask], u_tau[mask], atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL), f'{method}: tau differs')
        check(np.array_equal(panel['tail_warning'][:, :, mi][mask], ((l_tau < 10) | (u_tau < 10))[mask]), f'{method}: warning differs')
    if 'less_count' not in panel or 'equal_count' not in panel:
        errors.append('saved tie counts missing')
        return errors
    less, equal = (panel['less_count'], panel['equal_count'])
    check(less.shape == equal.shape == shape and less.dtype == equal.dtype == np.dtype('uint16'), 'tie count shape/dtype differs')
    if less.shape != shape or equal.shape != shape:
        return errors
    known = (less != UNKNOWN_COUNT) & (equal != UNKNOWN_COUNT)
    check(np.array_equal(less == UNKNOWN_COUNT, equal == UNKNOWN_COUNT), 'partial unknown tie count')
    check(np.all(less[known].astype(int) + equal[known].astype(int) <= B), 'tie count partition exceedsB')
    check(np.array_equal(less[:, :, 0], less[:, :, 1]) and np.array_equal(less[:, :, 0], less[:, :, 2]) and np.array_equal(equal[:, :, 0], equal[:, :, 1]) and np.array_equal(equal[:, :, 0], equal[:, :, 2]), 'methods do not share same tie counts')
    check(np.all(known[:, :, 1][valid[:, :, 1]]) and np.all(known[:, :, 2][valid[:, :, 2]]), 'valid bootstrap interval has unknown counts')
    bknown = known[:, :, 2]
    prob = (less[:, :, 2].astype(float) + 0.5 * equal[:, :, 2].astype(float)) / B
    expected_z = special.ndtri(prob[bknown])
    actual_z = panel['z0'][:, :, 2][bknown]
    check(np.array_equal(np.isposinf(expected_z), np.isposinf(actual_z)) and np.array_equal(np.isneginf(expected_z), np.isneginf(actual_z)) and np.allclose(actual_z[np.isfinite(expected_z)], expected_z[np.isfinite(expected_z)], atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL), 'BCa z0 differs from actual half-tie counts')
    return errors

def audit_stream(path, panel, dist, seed, inner):
    errors = []
    check = lambda condition, message: errors.append(message) if not condition else None
    with np.load(path, allow_pickle=False) as source:
        stream = {key: source[key] for key in source.files}
    indices, means = (stream['indices'], stream['means'])
    check(indices.shape == (B, N) and indices.dtype == np.dtype('uint8'), 'indices shape/dtype differ')
    check(means.shape == (4999,) and means.dtype == np.dtype('float64'), 'means shape/dtype differ')
    if errors:
        return errors
    check(np.all(indices < N), 'index outside source sample')
    if np.any(indices >= N):
        return errors
    for name, value in dict(distribution=dist, master_seed=seed, outer_id=0, budget_repeat_id=0, stream_id=inner, n=N, B=4999, R=20, streams_per_budget=120, alpha=0.05, namespace=MAIN_NAMESPACE, native_indices_dtype='int64', stored_indices_dtype='uint8').items():
        check(_meta(stream, name) == value, f'stream {name} metadata differ')
    digest = _meta(stream, 'sample_sha256')
    digest = digest.decode('ascii') if isinstance(digest, bytes) else str(digest)
    sample = panel['samples'][0]
    check(digest == sample_hash(sample), 'stream original hash differs')
    expected_means = np.mean(sample[indices], axis=1)
    check(np.array_equal(expected_means, means), 'saved means differ from indices/original')
    reference = reference_intervals(sample, expected_means)
    for mi, method in enumerate(METHODS):
        actual = {field: str(panel['reason_labels'][panel['reason_codes'][0, 0, mi, inner]]) if field == 'reason' else panel[field][0, 0, mi, inner].item() for field in reference[method]}
        errors.extend(compare_rows([dict(id=0, **reference[method])], [dict(id=0, **actual)], ['id'], label=method))
    return errors

def analytic_constants(distribution):
    """Unrounded accepted equations, independently evaluated without model imports."""
    if distribution == 'gamma':
        return dict(shape=1.0, scale=1.0)
    if distribution != 'lognormal':
        raise ValueError('unknown population')
    v = 2 * math.sinh(math.asinh(1) / 3)
    t = 1 + v ** 2
    s = math.sqrt(math.log1p(v ** 2))
    m = math.sqrt(t)
    d = m * v
    return dict(v=v, t=t, s=s, m=m, d=d)

def transform_raw(raw, distribution):
    raw = np.asarray(raw, dtype='<f8')
    if raw.shape[-1:] != (N,) or not np.isfinite(raw).all():
        raise ValueError('raw observations must be finite length5 rows')
    if distribution == 'gamma':
        if np.any(raw < 0):
            raise ValueError('Gamma raw observations outside nonnegative support')
        sample = raw - 1.0
    elif distribution == 'lognormal':
        constants = analytic_constants(distribution)
        with np.errstate(over='ignore', invalid='ignore'):
            sample = (np.exp(constants['s'] * raw) - constants['m']) / constants['d']
    else:
        raise ValueError('unknown population')
    if not np.isfinite(sample).all():
        raise ValueError('analytic transformation nonfinite')
    return np.asarray(sample, dtype='<f8')

def expected_original_key(distribution, outer, namespace=MAIN_NAMESPACE):
    return (namespace, {'gamma': 1, 'lognormal': 2}[distribution], N, outer, 0, 0)


def inspect_original_cell(cell, distribution, seed):
    errors = []
    check = lambda condition, message: errors.append(message) if not condition else None
    samples, raw = (cell['samples'], cell['raw'])
    shape = (10, N)
    check(samples.shape == raw.shape == shape and samples.dtype == raw.dtype == np.dtype('<f8'), 'original raw/sample shape or float64LE dtype differs')
    check(cell['outer_id'].dtype == np.dtype('uint16') and np.array_equal(cell['outer_id'], np.arange(10)), 'original IDs differ')
    for name, value in dict(distribution=distribution, master_seed=seed, n=N, namespace=MAIN_NAMESPACE, raw_kind='gamma_y' if distribution == 'gamma' else 'normal_z').items():
        check(_meta(cell, name) == value, f'original {name} differs')
    keys = np.array([expected_original_key(distribution, outer) for outer in range(10)], dtype=np.uint32)
    check(cell['original_keys'].dtype == np.dtype('uint32') and np.array_equal(cell['original_keys'], keys), 'original six-element keys differ')
    constants = analytic_constants(distribution)
    check(cell['constant_names'].astype(str).tolist() == list(constants), 'constant names/order differ')
    check(cell['constant_values'].dtype == np.dtype('<f8') and cell['constant_values'].shape == (len(constants),), 'constant dtype/shape differs')
    check(np.asarray(cell['constant_values'], dtype='<f8').tobytes() == np.array(list(constants.values()), dtype='<f8').tobytes(), 'unrounded constant bits differ')
    if errors:
        return errors
    try:
        transformed = transform_raw(raw, distribution)
    except ValueError as exc:
        return errors + [str(exc)]
    check(transformed.tobytes() == samples.tobytes(), 'raw analytic transform differs from saved sample bytes')
    for field, values in (('raw_sha256', raw), ('sample_sha256', samples)):
        check(cell[field].dtype == np.dtype('S64') and cell[field].shape == (10,), f'{field} dtype/shape differs')
        check(cell[field].astype(str).tolist() == [sample_hash(value) for value in values], f'{field} row hashes differ')
    if len({sample_hash(sample) for sample in samples}) != 10:
        errors.append('duplicate original arrays within new cell')
    return errors


def count_vectors(total=5, parts=5):
    """Every integer composition once; probabilities are multinomial, not uniform."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in count_vectors(total - first, parts - 1):
            yield ((first,) + tail)


def position_permutations(counts):
    """Distinct permutations of positions; duplicate observation values stay distinct."""
    remaining = list(counts)
    row = []
    size = sum(remaining)

    def visit():
        if len(row) == size:
            yield tuple(row)
            return
        for position, available in enumerate(remaining):
            if available:
                remaining[position] -= 1
                row.append(position)
                yield from visit()
                row.pop()
                remaining[position] += 1
    yield from visit()


def enumerate_by_composition(sample):
    """Evaluate every ordering, preserving accepted NumPy row-wise float64 means."""
    sample = np.asarray(sample, dtype='<f8')
    if sample.shape != (N,):
        raise ValueError('expected one saved n5 sample')
    rows = []
    weights = []
    compositions = []
    for composition in count_vectors():
        expected = math.factorial(5) // math.prod((math.factorial(value) for value in composition))
        permutations = list(position_permutations(composition))
        if len(permutations) != expected:
            raise ValueError('composition multiplicity differs')
        rows.extend(permutations)
        weights.append(expected)
        compositions.append(composition)
    if len(compositions) != 126 or len(rows) != ENUMERATED or sum(weights) != ENUMERATED or (len(set(rows)) != ENUMERATED):
        raise ValueError('enumeration is not all 3125 position tuples')
    indices = np.asarray(rows, dtype=np.uint8)
    with np.errstate(over='ignore', invalid='ignore'):
        means = np.mean(sample[indices], axis=1)
    code = np.zeros(ENUMERATED, dtype=np.int64)
    for column in range(N):
        code = code * 5 + indices[:, column]
    ordered = np.empty(ENUMERATED, dtype='<f8')
    ordered[code] = means
    support, frequency = np.unique(means, return_counts=True)
    frequency = frequency.astype(np.int64)
    return dict(indices=indices, means=means, ordered_means=ordered, support=support, frequency=frequency, cumulative=np.cumsum(frequency, dtype=np.int64), count_vectors=np.asarray(compositions, dtype=np.uint8), multinomial_weights=np.asarray(weights, dtype=np.int64))


def inverse_cdf(support, frequency, q):
    """Select a discrete atom using integer comparison with q's exact binary ratio."""
    support = np.asarray(support, dtype=float)
    frequency = np.asarray(frequency)
    if support.ndim != 1 or support.size == 0 or frequency.shape != support.shape or (frequency.dtype.kind not in 'iu') or np.any(frequency <= 0):
        raise ValueError('invalid discrete support/frequencies')
    if not np.isfinite(support).all() or np.any(support[1:] <= support[:-1]):
        raise ValueError('support not finite and strictly ordered')
    if isinstance(q,(bool,np.bool_)) or not np.isscalar(q):
        raise ValueError('q must be a scalar probability')
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('q outside [0,1]')
    cumulative = np.cumsum(frequency, dtype=np.int64)
    total = int(cumulative[-1])
    qratio = Fraction.from_float(float(q))
    selected = 0
    while selected < len(support) - 1 and int(cumulative[selected]) * qratio.denominator < qratio.numerator * total:
        selected += 1
    left = int(cumulative[selected - 1]) if selected else 0
    right = int(cumulative[selected])
    left_ratio = Fraction(left, total)
    right_ratio = Fraction(right, total)
    return dict(value=float(support[selected]), support_index=selected, frequency=int(frequency[selected]), left_count=left, right_count=right, cdf_left=float(left_ratio), cdf_right=float(right_ratio), q_minus_left=float(qratio - left_ratio), right_minus_q=float(right_ratio - qratio), q_numerator=str(qratio.numerator), q_denominator=str(qratio.denominator))


def adjusted_quantiles(z0, a):
    """Independent BCa branch, without clipping or finite-B z0 reuse."""
    if not np.isfinite(z0):
        return (np.full(2, np.nan), 'bca_bias_correction_nonfinite')
    if not np.isfinite(a):
        return (np.full(2, np.nan), 'bca_acceleration_undefined')
    z = special.ndtri(0.025)
    bounds = np.array([z, -z])
    den = 1 - a * (z0 + bounds)
    if not np.isfinite(den).all() or np.any(den <= 0):
        return (np.full(2, np.nan), 'bca_adjustment_singular')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        q = special.ndtr(z0 + (z0 + bounds) / den)
    if not np.isfinite(q).all() or not 0 <= q[0] < q[1] <= 1:
        return (q, 'bca_quantiles_invalid')
    return (q, '')






def joint_status(statuses):
    statuses=list(statuses)
    if 'not_supported' in statuses:
        return 'not_supported'
    return 'supported' if len(statuses)==2 and all(value=='supported' for value in statuses) else 'inconclusive'


def _npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def check_input_receipt(source, output, root=None):
    """Independently bind the read-only accepted files and preanalysis receipt."""
    root = ROOT if root is None else Path(root)
    source = Path(source)
    output = Path(output)
    manifest_path = output / 'manifest.json'
    if not manifest_path.exists():
        manifest_path = output.parent / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = json.loads((output / 'input_receipt.json').read_bytes())
    if manifest['accepted_head'] != ACCEPTED_M10 or receipt['accepted_head'] != ACCEPTED_M10:
        raise ValueError('accepted M10 head differs')
    if receipt['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('manifest receipt hash differs')
    if receipt['inputs'] != manifest['known_inputs'] or receipt['originals'] != manifest['original_ids_and_receipt']:
        raise ValueError('input receipt differs from frozen manifest')
    freeze = receipt['preanalysis_commit']
    if not isinstance(freeze, str) or len(freeze) != 40 or any((c not in '0123456789abcdef' for c in freeze)):
        raise ValueError('preanalysis commit is not a full Git SHA')
    for relative, expected in receipt['inputs'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe input receipt path')
        content = (root / path).read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\x00' + content).hexdigest()
        if expected != {'git_blob': blob, 'sha256': sha}:
            raise ValueError(f'accepted input hash differs: {relative}')
    source_prefix = source.relative_to(root)
    source_manifest = json.loads((source/'manifest.json').read_bytes())
    required = set(source_manifest['source_files_sha256'])
    for filename in ('manifest.json','originals/receipt.json','hypotheses.json','REPORT.md','resources.json','independent_audit.json'):
        required.add(str(source_prefix/filename))
    expected_names = {f'{dist}_seed{seed}.npz' for dist in DISTRIBUTIONS for seed in (0,1,2)}
    expected_streams = {f'{dist}_seed{seed}_outer0_repeat0_stream{stream}.npz' for dist in DISTRIBUTIONS for seed in (0,1,2) for stream in (0,29,30,59,60,119)}
    for folder,names in (('panels',expected_names),('originals',expected_names),('streams',expected_streams)):
        if {path.name for path in (source/folder).glob('*.npz')} != names:
            raise ValueError(f'accepted {folder} inventory differs')
        required.update(str(source_prefix/folder/name) for name in names)
    if set(manifest['known_inputs']) != required:
        raise ValueError('accepted input file inventory differs from the frozen source contract')
    original_receipt = json.loads((source / 'originals/receipt.json').read_bytes())
    if receipt['originals'] != original_receipt['records']:
        raise ValueError('accepted original receipt rows differ')
    if len(receipt['originals']) != 60:
        raise ValueError('planned sixty original receipt rows missing')
    if original_receipt['question_id']!='RQ-M10' or original_receipt['original_samples']!=60:
        raise ValueError('accepted original receipt design differs')
    if (manifest['n'], manifest['alpha'], manifest['finite_B'], manifest['ordered_n5'], manifest['series_per_original'], manifest['unique_saved_series']) != (5, 0.05, 4999, 3125, 2400, 144000):
        raise ValueError('frozen analysis dimensions differ')
    if manifest['new_draws'] != 0 or manifest['prng_replays'] != 0:
        raise ValueError('analysis contract permits unexpected draws')
    return (manifest, receipt, original_receipt)


def check_raw_link(source, dist, seed, panel, receipt):
    cell_path = Path(source) / 'originals' / f'{dist}_seed{seed}.npz'
    cell = _npz(cell_path)
    errors = inspect_original_cell(cell, dist, seed)
    if errors:
        raise ValueError('; '.join(errors))
    name = cell_path.name
    if receipt['original_files_sha256'][name] != hashlib.sha256(cell_path.read_bytes()).hexdigest():
        raise ValueError('accepted original file hash differs')
    receipt_digest = hashlib.sha256((Path(source) / 'originals/receipt.json').read_bytes()).hexdigest()
    if _meta(panel, 'original_receipt_sha256') != receipt_digest:
        raise ValueError('endpoint panel is not bound to raw receipt')
    if not np.array_equal(cell['samples'], panel['samples']) or cell['samples'].tobytes() != panel['samples'].tobytes():
        raise ValueError('raw and endpoint original sample bytes differ')
    if not np.array_equal(cell['sample_sha256'], panel['sample_sha256']):
        raise ValueError('raw and endpoint original hashes differ')
    records = {(row['distribution'], row['master_seed'], row['outer_id']): row for row in receipt['records']}
    if len(records) != 60:
        raise ValueError('duplicate/missing accepted receipt IDs')
    for outer in range(10):
        record = records[dist, seed, outer]
        if record['sample_sha256'] != sample_hash(cell['samples'][outer]) or record['raw_sha256'] != sample_hash(cell['raw'][outer]):
            raise ValueError('raw row receipt hash differs')
        if record['spawn_key'] != list(expected_original_key(dist, outer)) or record['entropy'] != seed or record['n'] != N or (record['namespace'] != MAIN_NAMESPACE):
            raise ValueError('accepted raw row key differs')
        if record['constants'] != analytic_constants(dist):
            raise ValueError('accepted raw row constants differ')
    return cell


def build_reference(sample):
    """Independent computer-conditional reference, including method-local invalidity."""
    sample = np.asarray(sample, dtype='<f8')
    if sample.shape != (N,) or not np.isfinite(sample).all():
        raise ValueError('reference requires finite n5 observations')
    enumerated = enumerate_by_composition(sample)
    means = enumerated['ordered_means']
    if not np.isfinite(means).all():
        raise ValueError('enumerated_means_nonfinite')
    theta = float(np.mean(sample))
    constant = bool(np.all(sample == sample[0]))
    se = float(_original_se(sample[None, :])[0])
    acceleration = np.nan if constant else jackknife_acceleration(sample)
    context_valid = bool(np.isfinite(theta) and np.isfinite(se) and se > 0)
    context_reason = 'sample_constant' if constant else 'sample_se_nonpositive_or_nonfinite'
    less = int(np.count_nonzero(means < theta))
    equal = int(np.count_nonzero(means == theta))
    probability = (less + .5 * equal) / ENUMERATED
    z0 = float(special.ndtri(probability))
    # Primary row order is a storage convention, not the audit enumeration method.
    codes = np.zeros(ENUMERATED, dtype=np.int64)
    for column in range(N):
        codes = codes * N + enumerated['indices'][:, column]
    indices = enumerated['indices'][np.argsort(codes)]
    support = enumerated['support'].copy()
    support[support == 0] = 0.0
    output = dict(indices=indices, means=means, support=support,
                  frequency=enumerated['frequency'], cumulative=enumerated['cumulative'],
                  theta=theta, se=se, a=acceleration, enum_less_count=less,
                  enum_equal_count=equal, p_star=probability, z0_star=z0,
                  methods=np.asarray(REFERENCE_METHODS), endpoints=np.asarray(ENDPOINTS),
                  n=N, enum_size=ENUMERATED, valid=np.zeros(2, dtype=bool),
                  reason=np.full(2, '', dtype='U96'))
    floating = ('q', 'E_cdf', 'E_linear', 'cdf_left', 'cdf_right', 'distance_left',
                'distance_right', 'tail_mass', 'enum_tail_count')
    integers = ('atom_index', 'atom_frequency', 'cdf_left_count', 'cdf_right_count')
    rationals = ('q_num', 'q_den', 'distance_left_num', 'distance_left_den',
                 'distance_right_num', 'distance_right_den')
    output.update({field: np.full((2, 2), np.nan) for field in floating})
    output.update({field: np.full((2, 2), -1, dtype=np.int64) for field in integers})
    output.update({field: np.full((2, 2), '', dtype='U400') for field in rationals})
    output['tail_warning'] = np.zeros((2, 2), dtype=bool)
    for method_index, method in enumerate(REFERENCE_METHODS):
        if method == 'bca':
            if not context_valid:
                output['reason'][method_index] = context_reason
                continue
            quantiles, reason = adjusted_quantiles(z0, acceleration)
            # An undefined branch leaves q unknown, matching the frozen writer.
            if reason and reason != 'bca_quantiles_invalid':
                output['reason'][method_index] = reason
                continue
        else:
            quantiles = np.array([.025, .975])
            reason = ''
        output['q'][method_index] = quantiles
        if reason:
            output['reason'][method_index] = reason
            continue
        for endpoint_index, q in enumerate(quantiles):
            atom = inverse_cdf(support, output['frequency'], q)
            q_fraction = Fraction(int(atom['q_numerator']), int(atom['q_denominator']))
            left_distance = q_fraction - Fraction(atom['left_count'], ENUMERATED)
            right_distance = Fraction(atom['right_count'], ENUMERATED) - q_fraction
            fields = dict(E_cdf=atom['value'], atom_index=atom['support_index'],
                          atom_frequency=atom['frequency'], cdf_left_count=atom['left_count'],
                          cdf_right_count=atom['right_count'], cdf_left=atom['cdf_left'],
                          cdf_right=atom['cdf_right'], distance_left=float(left_distance),
                          distance_right=float(right_distance), q_num=atom['q_numerator'],
                          q_den=atom['q_denominator'], distance_left_num=str(left_distance.numerator),
                          distance_left_den=str(left_distance.denominator),
                          distance_right_num=str(right_distance.numerator),
                          distance_right_den=str(right_distance.denominator),
                          E_linear=float(np.quantile(means, q, method='linear')),
                          tail_mass=min(q, 1-q), enum_tail_count=ENUMERATED*min(q, 1-q))
            for field, value in fields.items():
                output[field][method_index, endpoint_index] = value
            output['tail_warning'][method_index, endpoint_index] = fields['enum_tail_count'] < 10
        finite = np.isfinite(output['E_cdf'][method_index]).all()
        finite &= np.isfinite(output['E_linear'][method_index]).all()
        if finite:
            output['valid'][method_index] = True
        else:
            output['reason'][method_index] = 'interval_endpoints_nonfinite'
    return output


def saved_arrays(panel, outer):
    """Direct saved-data projection; r-major then k, without computation or RNG."""
    observed = panel['observed_streams'][outer].reshape(-1)
    result = dict(observed=observed, budget_repeat_id=np.repeat(np.arange(R, dtype=np.uint8), STREAMS),
                  stream_id=np.tile(np.arange(STREAMS, dtype=np.uint8), R))
    reasons = _reason_array(panel)[outer]
    for endpoint in ENDPOINTS:
        result['saved_t_'+endpoint] = panel[endpoint][outer, :, 0, :].reshape(-1)
    result['saved_t_valid'] = panel['valid'][outer, :, 0, :].reshape(-1)
    indices = (2, 1)
    result['saved_endpoint'] = np.stack([
        np.column_stack([panel[endpoint][outer, :, mi, :].reshape(-1) for endpoint in ENDPOINTS])
        for mi in indices], axis=1)
    result['saved_q'] = np.stack([
        np.column_stack([panel[key][outer, :, mi, :].reshape(-1) for key in ('q_low', 'q_high')])
        for mi in indices], axis=1)
    fields = dict(saved_valid='valid', saved_warning='tail_warning', saved_z0='z0', saved_a='a',
                  less_count='less_count', equal_count='equal_count',
                  saved_tau_low='tau_low', saved_tau_high='tau_high')
    for output_name, source_name in fields.items():
        result[output_name] = np.column_stack([panel[source_name][outer, :, mi, :].reshape(-1) for mi in indices])
    result['saved_reason'] = np.column_stack([reasons[:, mi, :].reshape(-1) for mi in indices])
    return result


def stream_comparison(reference, saved):
    endpoints = saved['saved_endpoint']
    scale = reference['se']
    method_mask = saved['observed'][:, None] & saved['saved_valid'] & reference['valid'][None, :]
    method_mask &= np.isfinite(endpoints).all(axis=2)
    method_mask &= endpoints[:, :, 0] <= endpoints[:, :, 1]
    quantiles = saved['saved_q']
    method_mask &= np.isfinite(quantiles).all(axis=2)
    method_mask &= (quantiles[:,:,0]>=0) & (quantiles[:,:,0]<quantiles[:,:,1]) & (quantiles[:,:,1]<=1)
    method_mask[:, 0] &= np.isfinite(saved['saved_z0'][:, 0]) & np.isfinite(saved['saved_a'][:, 0])
    method_mask &= bool(np.isfinite(scale) and scale > 0)
    endpoint_mask = np.repeat(method_mask[:, :, None], 2, axis=2)
    differences = {}
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        differences['delta'] = np.where(endpoint_mask, (endpoints-reference['E_cdf'])/scale, np.nan)
        differences['linear_delta'] = np.where(endpoint_mask, (endpoints-reference['E_linear'])/scale, np.nan)
        differences['convention_offset'] = np.where(endpoint_mask, (reference['E_linear']-reference['E_cdf'])/scale, np.nan)
    for values in differences.values():
        endpoint_mask &= np.isfinite(values)
    for values in differences.values():
        values[~endpoint_mask] = np.nan
    return {**saved, **differences, 'error_valid': endpoint_mask,
            'decomposition_residual': differences['delta']-(differences['linear_delta']+differences['convention_offset'])}


METRIC_FIELDS = ('b', 'variance_ddof0', 'sd_ddof0', 'rmse', 'mean_absolute_error',
                 'abs_p50', 'abs_p95', 'abs_max', 'linear_mean', 'convention_offset',
                 'linear_mse', 'convention_mse', 'cross_term', 'mse',
                 'rmse_identity_residual', 'rmse_identity_scale', 'rmse_identity_normalized',
                 'mse_decomposition_residual', 'mse_decomposition_scale',
                 'mse_decomposition_normalized', 'endpoint_decomposition_max_abs')
REFERENCE_ARRAY_FIELDS = ('q', 'E_cdf', 'E_linear', 'atom_index', 'atom_frequency',
                          'cdf_left_count', 'cdf_right_count', 'cdf_left', 'cdf_right',
                          'distance_left', 'distance_right', 'q_num', 'q_den',
                          'distance_left_num', 'distance_left_den', 'distance_right_num',
                          'distance_right_den', 'tail_mass', 'enum_tail_count', 'tail_warning')


def error_moments(delta, linear, offset):
    """Direct MSE, both cross terms, and explicit ddof0 moment identities."""
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        bias = float(np.mean(delta))
        variance = float(np.mean((delta-bias)**2))
        mse = float(np.mean(delta**2))
        rmse = float(np.sqrt(mse))
        absolute = np.abs(delta)
        linear_mse = float(np.mean(linear**2))
        convention_mse = float(np.mean(offset**2))
        cross = float(2*np.mean(linear*offset))
        identity = rmse*rmse-(bias*bias+variance)
        decomposition = mse-(linear_mse+convention_mse+cross)
    identity_scale = max(abs(mse), abs(bias*bias+variance))
    decomposition_scale = max(abs(mse), abs(linear_mse+convention_mse+cross))
    endpoint_residual = float(np.max(np.abs(delta-(linear+offset))))
    endpoint_scale = max(1.0, float(np.max(np.abs(delta))))
    checks = ((identity, identity_scale), (decomposition, decomposition_scale), (endpoint_residual, endpoint_scale))
    identity_ok = all(abs(residual) <= IDENTITY_ATOL+IDENTITY_RTOL*scale for residual, scale in checks)
    return dict(b=bias, variance_ddof0=variance, sd_ddof0=float(np.sqrt(variance)), rmse=rmse,
                mean_absolute_error=float(absolute.mean()), abs_p50=float(np.quantile(absolute,.5,method='linear')),
                abs_p95=float(np.quantile(absolute,.95,method='linear')), abs_max=float(absolute.max()),
                linear_mean=float(np.mean(linear)), convention_offset=float(np.mean(offset)),
                linear_mse=linear_mse, convention_mse=convention_mse, cross_term=cross, mse=mse,
                rmse_identity_residual=identity, rmse_identity_scale=identity_scale,
                rmse_identity_normalized=identity/max(1.,identity_scale),
                mse_decomposition_residual=decomposition, mse_decomposition_scale=decomposition_scale,
                mse_decomposition_normalized=decomposition/max(1.,decomposition_scale),
                endpoint_decomposition_max_abs=endpoint_residual, identity_ok=bool(identity_ok))


def endpoint_metrics(comparison, method_index, endpoint_index):
    valid = comparison['error_valid'][:, method_index, endpoint_index]
    count = int(valid.sum())
    output = dict(planned_series=2400, valid_series=count,
                  missing_series=int((~comparison['observed']).sum()),
                  invalid_series=int((comparison['observed'] & ~valid).sum()),
                  complete=count == 2400, identity_ok=False)
    output.update({field: np.nan for field in METRIC_FIELDS})
    if not count:
        output.update({'conditional_'+field: np.nan for field in METRIC_FIELDS})
        output['conditional_identity_ok'] = False
        return output
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        moments = error_moments(comparison['delta'][valid, method_index, endpoint_index],
                                comparison['linear_delta'][valid, method_index, endpoint_index],
                                comparison['convention_offset'][valid, method_index, endpoint_index])
    output.update({'conditional_'+field: value for field, value in moments.items()})
    numerical = all(np.isfinite(moments[field]) for field in METRIC_FIELDS)
    if numerical and not moments['identity_ok']:
        raise ValueError('independent RMSE/decomposition identity failed')
    if not numerical:
        output.update(complete=False, reason='derived_metric_nonfinite')
    elif output['complete']:
        output.update(moments)
    return output


def classify_original(endpoint_rows):
    if len(endpoint_rows) != 2:
        raise ValueError('two endpoint rows required')
    complete = all(row['complete'] and row['identity_ok'] for row in endpoint_rows)
    error = max(row['rmse'] for row in endpoint_rows) if complete else np.nan
    missing = any(row['missing_series'] for row in endpoint_rows)
    invalid = any(row['invalid_series'] for row in endpoint_rows) or (not complete and not missing)
    return dict(complete=bool(complete), e=float(error),
                pass_flag=bool(error <= .10) if complete else None,
                margin=float(.10-error) if complete else np.nan,
                missing=missing, invalid=invalid,
                reason='' if complete else 'required_reference_scale_or_series_invalid_or_missing')


def group_decision(originals):
    identities = {(row['master_seed'], row['outer_id']) for row in originals}
    if len(identities) != len(originals):
        raise ValueError('duplicate original identities')
    if any(seed not in (0,1,2) or outer not in range(10) for seed,outer in identities):
        raise ValueError('original identity outside fixed design')
    complete = [row for row in originals if row['complete']]
    passed = sum(row['pass_flag'] is True for row in complete)
    total = len(complete)
    all_complete = len(originals) == total == 30
    return dict(planned_N=30, observed_N=len(originals), complete_N=total,
                pass_count=passed, fail_count=total-passed, unknown_count=30-total,
                invalid_N=sum(bool(row['invalid']) for row in originals),
                missing_N=30-len(originals)+sum(bool(row['missing']) for row in originals),
                required_pass_count=27,
                status=('supported' if passed>=27 else 'not_supported') if all_complete else 'inconclusive',
                pass_fraction=passed/30 if all_complete else np.nan,
                conditional_pass_fraction=passed/total if total else np.nan)


def rank_and_represent(rows):
    examples = []
    for distribution in DISTRIBUTIONS:
        for method in REFERENCE_METHODS:
            complete = [row for row in rows if row['distribution']==distribution and row['method']==method and row['complete']]
            ascending = sorted(complete, key=lambda row:(row['e'], row['master_seed'], row['outer_id']))
            for rank, row in enumerate(ascending, 1):
                row['rank_e'] = rank
            if method != 'bca' or not ascending:
                continue
            smallest = ascending[0]
            largest = min(ascending, key=lambda row:(-row['e'], row['master_seed'], row['outer_id']))
            for selection, row in (('minimum', smallest), ('maximum', largest)):
                example = dict(selection=selection)
                example.update({key:row[key] for key in ('distribution','method','master_seed','outer_id','sample_sha256','e','margin','rank_e')})
                examples.append(example)
    return examples


def summary_rows(original_rows):
    rows = []
    for distribution in DISTRIBUTIONS:
        for method in REFERENCE_METHODS:
            group = [row for row in original_rows if row['distribution']==distribution and row['method']==method]
            complete = [row for row in group if row['complete']]
            output = dict(distribution=distribution, method=method, **group_decision(group))
            for prefix, selected in (('',group if len(group)==len(complete)==30 else []),('conditional_',complete)):
                values = np.asarray([row['e'] for row in selected])
                statistics = dict(e_mean=float(np.mean(values)) if values.size else np.nan,
                                  e_p50=float(np.quantile(values,.5,method='linear')) if values.size else np.nan,
                                  e_p95=float(np.quantile(values,.95,method='linear')) if values.size else np.nan,
                                  e_max=float(np.max(values)) if values.size else np.nan)
                output.update({prefix+key:value for key,value in statistics.items()})
            output['primary_hypothesis'] = method == 'bca'
            if method != 'bca':
                output['status'] = 'descriptive_only'
            rows.append(output)
    return rows


def hypothesis_reference(summary):
    groups = {row['distribution']:{key:value for key,value in row.items() if key not in ('distribution','method')}
              for row in summary if row['method']=='bca'}
    status = joint_status([groups[distribution]['status'] for distribution in DISTRIBUTIONS])
    return dict(question_id='RQ-M11', hypothesis='H11-E', target_method='bca', RMSE_limit=.10,
                required_fraction=.90, required_count=27, planned_N_per_distribution=30,
                original_n=5, series_per_original=2400,
                reference='conditional_binary64_ordered_enumeration_inverseCDF',
                computed_status=status, status=status, groups=groups, verification_required=True,
                interpretation='Fixed-data descriptive classification. Public conclusion requires completed run and passed independent audit; otherwise hold. No population90% guarantee or new t interval/pvalue.')


def assert_arrays(expected, actual, label, exact_fields=()):
    """Every persisted field is checked; branch/support/IDs are never allclose."""
    if set(expected) != set(actual):
        missing = sorted(set(expected)-set(actual))
        extra = sorted(set(actual)-set(expected))
        raise ValueError(f'{label}: field schema differs; missing={missing}, extra={extra}')
    for field, values in expected.items():
        target = np.asarray(values)
        observed = actual[field]
        if observed.dtype.hasobject or target.shape != observed.shape:
            raise ValueError(f'{label}/{field}: shape/object dtype differs')
        if target.dtype.kind in 'biuUS' or field in exact_fields:
            equal = np.array_equal(target, observed, equal_nan=True) if target.dtype.kind=='f' else np.array_equal(target, observed)
        else:
            equal = np.allclose(target, observed, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL, equal_nan=True)
        if not equal:
            raise ValueError(f'{label}/{field}: values differ')
        # Dtype categories are scientific storage contracts; Unicode width alone
        # is not scientific, whereas integer precision and binary64 are fixed.
        if target.dtype.kind=='f' and observed.dtype!=np.dtype('float64'):
            raise ValueError(f'{label}/{field}: non-binary64 numeric storage')
        if target.dtype.kind in 'biu' and observed.dtype!=target.dtype:
            raise ValueError(f'{label}/{field}: mask/integer storage dtype differs')
        if target.dtype.kind in 'US' and observed.dtype.kind!=target.dtype.kind:
            raise ValueError(f'{label}/{field}: text storage dtype differs')


def _rows_for_original(panel, outer, reference, comparison):
    identity = dict(distribution=_meta(panel,'distribution'), master_seed=int(_meta(panel,'master_seed')),
                    outer_id=outer, sample_sha256=sample_hash(panel['samples'][outer]))
    references, endpoints, originals = [], [], []
    for mi, method in enumerate(REFERENCE_METHODS):
        local = []
        for ei, endpoint in enumerate(ENDPOINTS):
            record = dict(**identity, method=method, endpoint=endpoint, theta=reference['theta'],
                          se=reference['se'], a=reference['a'], enum_less_count=reference['enum_less_count'],
                          enum_equal_count=reference['enum_equal_count'], p_star=reference['p_star'],
                          z0_star=reference['z0_star'] if method=='bca' else np.nan,
                          support_count=len(reference['support']), reference_valid=bool(reference['valid'][mi]),
                          reference_reason=str(reference['reason'][mi]))
            record.update({field:reference[field][mi,ei].item() for field in REFERENCE_ARRAY_FIELDS})
            references.append(record)
            measured = endpoint_metrics(comparison,mi,ei)
            finite_q = comparison['saved_q'][:,mi,ei]
            finite_q = finite_q[np.isfinite(finite_q) & comparison['observed']]
            measured['saved_q_min'] = float(np.min(finite_q)) if finite_q.size else np.nan
            measured['saved_q_max'] = float(np.max(finite_q)) if finite_q.size else np.nan
            endpoints.append(dict(**identity,method=method,endpoint=endpoint,**measured))
            local.append(measured)
        originals.append(dict(**identity, method=method, **classify_original(local), rank_e=None,
                              saved_warning_series=int(np.count_nonzero(comparison['observed'] & comparison['saved_warning'][:,mi])),
                              reference_valid=bool(reference['valid'][mi]), reference_reason=str(reference['reason'][mi])))
    return references, endpoints, originals


def audit(source: Path, output: Path):
    """Audit saved M11 results without writes, primary imports, or RNG calls.

    Root's guard owns the actual invocation, resource accounting, prerequisite
    commit verification, and exclusive persistence of the returned report.
    """
    source, output = Path(source), Path(output)
    manifest, input_receipt, original_receipt = check_input_receipt(source,output)
    reference_rows, endpoint_rows, original_rows = [], [], []
    input_hashes, output_hashes = {}, {}
    panels = []
    originals_seen = set()
    # Validate all sixty raw inputs and every saved diagnostic before enumerating
    # any fixture/real original. There is no early scientific selection.
    for distribution in DISTRIBUTIONS:
        for seed in (0,1,2):
            filename = f'{distribution}_seed{seed}.npz'
            panel_path = source/'panels'/filename
            panel = _npz(panel_path)
            errors = inspect_panel(panel)
            if errors:
                raise ValueError(f'{filename}: '+'; '.join(errors))
            check_raw_link(source,distribution,seed,panel,original_receipt)
            for sample in panel['samples']:
                digest = sample_hash(sample)
                if digest in originals_seen:
                    raise ValueError('duplicate original across accepted input cells')
                originals_seen.add(digest)
            panels.append(panel)
            input_hashes[str(panel_path.relative_to(source))] = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    expected_stems = {f'{dist}_seed{seed}_outer{outer}.npz' for dist in DISTRIBUTIONS for seed in (0,1,2) for outer in range(10)}
    for folder in ('references','comparisons'):
        actual_stems = {path.name for path in (output/folder).glob('*.npz')}
        if actual_stems != expected_stems:
            raise ValueError(f'{folder}: fixed sixty saved original files differ')
    finite_streams = 0
    for panel in panels:
        dist = _meta(panel,'distribution')
        seed = int(_meta(panel,'master_seed'))
        for outer in range(10):
            reference = build_reference(panel['samples'][outer])
            saved = saved_arrays(panel,outer)
            comparison = stream_comparison(reference,saved)
            identity = dict(distribution=dist, master_seed=seed, outer_id=outer,
                            sample_sha256=sample_hash(panel['samples'][outer]))
            metadata = dict(**identity, methods=np.asarray(REFERENCE_METHODS), endpoints=np.asarray(ENDPOINTS),
                            source_question_id='RQ-M10', question_id='RQ-M11', n=N, B=B, R=R, K=STREAMS)
            stem = f'{dist}_seed{seed}_outer{outer}.npz'
            reference_path = output/'references'/stem
            comparison_path = output/'comparisons'/stem
            assert_arrays({**reference,**metadata},_npz(reference_path),stem+'/reference',
                          exact_fields=('indices','means','support','frequency','cumulative','E_cdf'))
            assert_arrays({**comparison,**metadata},_npz(comparison_path),stem+'/comparison',exact_fields=tuple(saved))
            for path in (reference_path,comparison_path):
                output_hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
            local = _rows_for_original(panel,outer,reference,comparison)
            reference_rows.extend(local[0])
            endpoint_rows.extend(local[1])
            original_rows.extend(local[2])
        for stream in (0,29,30,59,60,119):
            path = source/'streams'/f'{dist}_seed{seed}_outer0_repeat0_stream{stream}.npz'
            errors = audit_stream(path,panel,dist,seed,stream)
            if errors:
                raise ValueError(f'{path.name}: '+'; '.join(errors))
            finite_streams += 1
            input_hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    representatives = rank_and_represent(original_rows)
    summaries = summary_rows(original_rows)
    hypotheses = hypothesis_reference(summaries)
    tables = (('reference_metrics.csv',reference_rows,('distribution','master_seed','outer_id','method','endpoint')),
              ('endpoint_metrics.csv',endpoint_rows,('distribution','master_seed','outer_id','method','endpoint')),
              ('original_metrics.csv',original_rows,('distribution','master_seed','outer_id','method')),
              ('summary.csv',summaries,('distribution','method')))
    for filename, expected, keys in tables:
        observed_rows = _csv(output/filename)
        expected_order = [tuple(str(row[key]) for key in keys) for row in expected]
        observed_order = [tuple(str(row[key]) for key in keys) for row in observed_rows]
        if expected_order != observed_order:
            raise ValueError(f'{filename}: row identity/order differs')
        csv_expected = [{key:(None if isinstance(value,(float,np.floating)) and not np.isfinite(value) else value)
                         for key,value in row.items()} for row in expected]
        errors = compare_rows(csv_expected,observed_rows,keys,label=filename)
        if errors:
            raise ValueError('; '.join(errors[:30]))
        output_hashes[filename] = hashlib.sha256((output/filename).read_bytes()).hexdigest()
    saved_hypotheses = json.loads((output/'hypotheses.json').read_bytes())
    errors = compare_json(hypotheses,saved_hypotheses)
    expected_examples = dict(rule='BCa perpopulation min/max e; ties seed,outer ascending',examples=representatives)
    errors.extend(compare_json(expected_examples,json.loads((output/'representatives.json').read_bytes()),location='representatives'))
    if errors:
        raise ValueError('; '.join(errors[:30]))
    resources = json.loads((output/'analysis_resources.json').read_bytes())
    identities = [dict(distribution=_meta(panel,'distribution'),master_seed=int(_meta(panel,'master_seed')),
                       outer_id=outer,sample_sha256=sample_hash(panel['samples'][outer]))
                  for panel in panels for outer in range(10)]
    required = dict(question_id='RQ-M11', completed_originals=60, completed_ids=identities,
                    new_originals=0,new_random_bootstrap_streams=0,PRNG_replays=0,
                    enumerated_ordered_rows=187500,enumerated_indices=937500,
                    saved_common_series=144000,status='completed',
                    scope='Internal deterministic enumeration/comparison/output loop only; external guard owns full180second/RSS/50MiB measurements.')
    errors = compare_json(required,resources,location='analysis_resources')
    elapsed = resources.get('analysis_loop_seconds')
    if errors or not isinstance(elapsed,(int,float)) or not np.isfinite(elapsed) or elapsed<0:
        raise ValueError('analysis completion/resource metadata differs: '+'; '.join(errors))
    json_hashes = {name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                   for name in ('hypotheses.json','representatives.json','analysis_resources.json','input_receipt.json')}
    return json_safe(dict(status='passed',mismatch_count=0,question_id='RQ-M11',
                originals=60,ordered_rows=187500,conditional_indices=937500,
                saved_common_streams=144000,saved_method_endpoints=576000,
                reconstructed_finite_streams=finite_streams,
                table_counts={filename:len(rows) for filename,rows,_ in tables},
                groups={dist:groups for dist,groups in hypotheses['groups'].items()},
                computed_status=hypotheses['computed_status'],preanalysis_commit=input_receipt['preanalysis_commit'],
                input_panel_and_stream_sha256=input_hashes,output_npz_and_csv_sha256=output_hashes,
                output_json_sha256=json_hashes,
                limitations=['No RNG regenerated or replayed. Only the existing36 saved index streams were reconstructed; the remaining143964 generation processes were not reconstructed.',
                             'The 126 count vectors with all distinct position permutations reconstruct the finite binary64 conditional distribution, not real-arithmetic or population truth.',
                             'Shared primitives: NumPy binary64 row means/linear quantiles and SciPy special functions; no primary enumeration, aggregation or decision imports.',
                             'This is implementation cross-checking, not external peer review, PRNG-independence proof, or a new population confidence interval.']))


def json_safe(value):
    """Returned reports use strict JSON nulls for undefined descriptive fields."""
    if isinstance(value,dict):
        return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value,np.generic):
        value=value.item()
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value
