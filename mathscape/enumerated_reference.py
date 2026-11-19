"""RQ-M11 deterministic finite empirical reference; no sampling or RNG.

The reference preserves all ordered binary64 reductions, including permutation
rounding differences. It is a conditional computer distribution, not a
population truth or unconditional B->infinity limit.
"""
from fractions import Fraction
import numpy as np
from scipy.special import ndtr, ndtri
METHODS = ('bca', 'percentile')
ENDPOINTS = ('lower', 'upper')
N = 5
ENUM_SIZE = 3125
SAVED_SERIES = 2400
NUMERIC_RTOL, NUMERIC_ATOL = (2e-10, 2e-12)
IDENTITY_RTOL = IDENTITY_ATOL = 2e-12
RMSE_LIMIT = 0.1
PASS_REQUIRED = 27

def ordered_indices():
    """Lexicographic base5: last index changes fastest, positions stay distinct."""
    return (np.arange(ENUM_SIZE, dtype=np.int64)[:, None] // 5 ** np.arange(4, -1, -1) % 5).astype(np.uint8)

def _sample(sample):
    raw = np.asarray(sample)
    if raw.shape != (N,) or raw.dtype.kind not in 'iuf':
        raise ValueError('sample must be a real length5 vector')
    values = np.asarray(raw, dtype='<f8', order='C')
    if not np.isfinite(values).all():
        raise ValueError('sample_nonfinite')
    return values

def enumerate_distribution(sample):
    """Evaluate each ordered resample with the exact M10 row reduction."""
    sample = _sample(sample)
    indices = ordered_indices()
    with np.errstate(over='ignore', invalid='ignore'):
        means = np.mean(sample[indices], axis=1)
    if not np.isfinite(means).all():
        raise ValueError('enumerated_means_nonfinite')
    support, frequency = np.unique(means, return_counts=True)
    support[support == 0] = 0.0
    return dict(indices=indices, means=np.asarray(means, dtype='<f8'), support=np.asarray(support, dtype='<f8'), frequency=frequency.astype(np.int64), cumulative=np.cumsum(frequency, dtype=np.int64))

def inverse_cdf(support, frequency, q):
    """First cumulative mass >= exact binary64 q, using integer comparisons."""
    support = np.asarray(support)
    frequency = np.asarray(frequency)
    if support.ndim != 1 or not len(support) or frequency.shape != support.shape or (support.dtype.kind not in 'iuf') or (frequency.dtype.kind not in 'iu'):
        raise ValueError('invalid CDF arrays')
    if not np.isfinite(support).all() or np.any(np.diff(support) <= 0) or np.any(frequency <= 0):
        raise ValueError('invalid support/frequency')
    if isinstance(q, (bool, np.bool_)) or not np.isscalar(q):
        raise ValueError('q must be a finite probability')
    q = float(q)
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('q must lie in[0,1]')
    total = sum(map(int, frequency))
    qn, qd = q.as_integer_ratio()
    cumulative = np.cumsum(frequency, dtype=np.int64)
    atom = 0 if q == 0 else next((i for i, count in enumerate(cumulative) if int(count) * qd >= qn * total))
    left = int(cumulative[atom - 1]) if atom else 0
    right = int(cumulative[atom])
    dl = Fraction(qn, qd) - Fraction(left, total)
    dr = Fraction(right, total) - Fraction(qn, qd)
    return dict(
        value=float(support[atom]), atom_index=atom,
        atom_frequency=int(frequency[atom]), cdf_left_count=left,
        cdf_right_count=right, cdf_left=left / total, cdf_right=right / total,
        q_num=str(qn), q_den=str(qd),
        distance_left_num=str(dl.numerator), distance_left_den=str(dl.denominator),
        distance_right_num=str(dr.numerator), distance_right_den=str(dr.denominator),
        distance_left=float(dl), distance_right=float(dr),
    )

def sample_context(sample):
    values = _sample(sample)
    theta = float(values.mean())
    if np.all(values == values[0]):
        return dict(theta=theta, se=np.nan, a=np.nan, valid=False, reason='sample_constant')
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        se = float(values.std(ddof=1) / np.sqrt(N))
        centered = values - theta
        normalized = centered / np.max(np.abs(centered))
        a = float(np.sum(normalized ** 3) / (6 * np.sum(normalized ** 2) ** 1.5))
    valid = bool(np.isfinite(theta) and np.isfinite(se) and (se > 0))
    return dict(theta=theta, se=se, a=a, valid=valid, reason='' if valid else 'sample_se_nonpositive_or_nonfinite')

def build_reference(sample):
    """CDF and linear references share enumeration q*, never finiteB q/z0."""
    values = _sample(sample)
    distribution = enumerate_distribution(values)
    context = sample_context(values)
    means = distribution['means']
    theta = context['theta']
    less = int(np.count_nonzero(means < theta))
    equal = int(np.count_nonzero(means == theta))
    probability = (less + 0.5 * equal) / ENUM_SIZE
    z0 = float(ndtri(probability))
    out = {
        **distribution, **context, 'enum_less_count': less,
        'enum_equal_count': equal, 'p_star': probability, 'z0_star': z0,
        'methods': np.array(METHODS), 'endpoints': np.array(ENDPOINTS),
        'n': N, 'enum_size': ENUM_SIZE, 'valid': np.zeros(2, bool),
        'reason': np.full(2, '', dtype='U96'),
    }
    float_fields = ('q', 'E_cdf', 'E_linear', 'cdf_left', 'cdf_right', 'distance_left', 'distance_right', 'tail_mass', 'enum_tail_count')
    integer_fields = ('atom_index', 'atom_frequency', 'cdf_left_count', 'cdf_right_count')
    rational_fields = ('q_num', 'q_den', 'distance_left_num', 'distance_left_den', 'distance_right_num', 'distance_right_den')
    out.update({k: np.full((2, 2), np.nan) for k in float_fields})
    out.update({k: np.full((2, 2), -1, np.int64) for k in integer_fields})
    out.update({k: np.full((2, 2), '', dtype='U400') for k in rational_fields})
    out['tail_warning'] = np.zeros((2, 2), bool)
    bounds = np.array([ndtri(0.025), -ndtri(0.025)])
    for mi, method in enumerate(METHODS):
        q = np.array([0.025, 0.975])
        if method == 'bca':
            if not context['valid']:
                out['reason'][mi] = context['reason']
                continue
            if not np.isfinite(z0):
                out['reason'][mi] = 'bca_bias_correction_nonfinite'
                continue
            if not np.isfinite(context['a']):
                out['reason'][mi] = 'bca_acceleration_undefined'
                continue
            denominator = 1 - context['a'] * (z0 + bounds)
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                out['reason'][mi] = 'bca_adjustment_singular'
                continue
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                q = ndtr(z0 + (z0 + bounds) / denominator)
        out['q'][mi] = q
        if not np.isfinite(q).all() or not 0 <= q[0] < q[1] <= 1:
            out['reason'][mi] = 'bca_quantiles_invalid'
            continue
        for ei, prob in enumerate(q):
            selected = inverse_cdf(out['support'], out['frequency'], prob)
            out['E_cdf'][mi, ei] = selected.pop('value')
            for k, v in selected.items():
                out[k][mi, ei] = v
            out['E_linear'][mi, ei] = np.quantile(means, prob, method='linear')
            out['tail_mass'][mi, ei] = min(prob, 1 - prob)
            out['enum_tail_count'][mi, ei] = ENUM_SIZE * min(prob, 1 - prob)
            out['tail_warning'][mi, ei] = out['enum_tail_count'][mi, ei] < 10
        if not np.isfinite(out['E_cdf'][mi]).all() or not np.isfinite(out['E_linear'][mi]).all():
            out['reason'][mi] = 'interval_endpoints_nonfinite'
            continue
        out['valid'][mi] = True
    return out

def stream_comparison(reference, saved):
    """Saved series order r*120+k; method order BCa/percentile, then lower/upper."""
    ends = np.asarray(saved['saved_endpoint'], dtype=float)
    if ends.shape != (SAVED_SERIES, 2, 2):
        raise ValueError('exactly2400series x2methods x2endpoints required')
    valid = np.asarray(saved['saved_valid'])
    observed = np.asarray(saved['observed'])
    if valid.shape != (SAVED_SERIES, 2) or valid.dtype.kind != 'b' or observed.shape != (SAVED_SERIES,) or (observed.dtype.kind != 'b'):
        raise ValueError('invalid saved masks')
    scale = reference['se']
    scale_valid = np.isfinite(scale) and scale > 0
    mask = observed[:, None] & valid & np.asarray(reference['valid'])[None, :]
    mask &= np.isfinite(ends).all(axis=2) & (ends[:, :, 0] <= ends[:, :, 1])
    saved_q = np.asarray(saved['saved_q'])
    mask &= np.isfinite(saved_q).all(axis=2)
    mask &= ((0 <= saved_q[:, :, 0]) &
             (saved_q[:, :, 0] < saved_q[:, :, 1]) &
             (saved_q[:, :, 1] <= 1))
    mask[:, 0] &= np.isfinite(saved['saved_z0'][:, 0]) & np.isfinite(saved['saved_a'][:, 0])
    mask &= scale_valid
    endpoint_mask = np.broadcast_to(mask[:, :, None], ends.shape).copy()
    delta = np.full(ends.shape, np.nan)
    linear = np.full(ends.shape, np.nan)
    offset = np.full(ends.shape, np.nan)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        np.divide(ends - reference['E_cdf'], scale, out=delta, where=endpoint_mask)
        np.divide(ends - reference['E_linear'], scale, out=linear, where=endpoint_mask)
        np.divide(np.broadcast_to(reference['E_linear'] - reference['E_cdf'], ends.shape), scale, out=offset, where=endpoint_mask)
    endpoint_mask &= np.isfinite(delta) & np.isfinite(linear) & np.isfinite(offset)
    for array in (delta, linear, offset):
        array[~endpoint_mask] = np.nan
    return {**saved, 'error_valid': endpoint_mask, 'delta': delta, 'linear_delta': linear, 'convention_offset': offset, 'decomposition_residual': delta - (linear + offset)}

def _moments(delta, linear, offset):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        b = float(np.mean(delta))
        v = float(np.mean((delta - b) ** 2))
        mse = float(np.mean(delta ** 2))
        r = float(np.sqrt(mse))
    absolute = np.abs(delta)
    linear_mse = float(np.mean(linear ** 2))
    offset_mse = float(np.mean(offset ** 2))
    cross = float(2 * np.mean(linear * offset))
    identity = r * r - (b * b + v)
    decomp = mse - (linear_mse + offset_mse + cross)
    scale = max(abs(mse), abs(b * b + v))
    dscale = max(abs(mse), abs(linear_mse + offset_mse + cross))
    residual = float(np.max(np.abs(delta - (linear + offset))))
    ok = abs(identity) <= IDENTITY_ATOL + IDENTITY_RTOL * scale and abs(decomp) <= IDENTITY_ATOL + IDENTITY_RTOL * dscale and (residual <= IDENTITY_ATOL + IDENTITY_RTOL * max(1.0, float(np.max(np.abs(delta)))))
    return dict(
        b=b, variance_ddof0=v, sd_ddof0=float(np.sqrt(v)), rmse=r,
        mean_absolute_error=float(absolute.mean()),
        abs_p50=float(np.quantile(absolute, 0.5, method='linear')),
        abs_p95=float(np.quantile(absolute, 0.95, method='linear')),
        abs_max=float(absolute.max()), linear_mean=float(linear.mean()),
        convention_offset=float(offset.mean()), linear_mse=linear_mse,
        convention_mse=offset_mse, cross_term=cross, mse=mse,
        rmse_identity_residual=identity, rmse_identity_scale=scale,
        rmse_identity_normalized=identity / max(1.0, scale),
        mse_decomposition_residual=decomp, mse_decomposition_scale=dscale,
        mse_decomposition_normalized=decomp / max(1.0, dscale),
        endpoint_decomposition_max_abs=residual, identity_ok=bool(ok),
    )
METRIC_FIELDS = (
    'b', 'variance_ddof0', 'sd_ddof0', 'rmse', 'mean_absolute_error',
    'abs_p50', 'abs_p95', 'abs_max', 'linear_mean', 'convention_offset',
    'linear_mse', 'convention_mse', 'cross_term', 'mse',
    'rmse_identity_residual', 'rmse_identity_scale', 'rmse_identity_normalized',
    'mse_decomposition_residual', 'mse_decomposition_scale',
    'mse_decomposition_normalized', 'endpoint_decomposition_max_abs',
)

def endpoint_metrics(comparison, mi, ei):
    valid = np.asarray(comparison['error_valid'][:, mi, ei])
    count = int(valid.sum())
    complete = count == SAVED_SERIES
    out = dict(planned_series=SAVED_SERIES, valid_series=count, missing_series=int((~comparison['observed']).sum()), invalid_series=int((comparison['observed'] & ~valid).sum()), complete=complete, identity_ok=False)
    out.update({k: np.nan for k in METRIC_FIELDS})
    if count:
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            measured = _moments(
                comparison['delta'][valid, mi, ei],
                comparison['linear_delta'][valid, mi, ei],
                comparison['convention_offset'][valid, mi, ei],
            )
        out.update({'conditional_' + k: v for k, v in measured.items()})
        numerical = all((np.isfinite(measured[field]) for field in METRIC_FIELDS))
        if numerical and (not measured['identity_ok']):
            raise ArithmeticError('RMSE/decomposition identity failed; scientific classification forbidden')
        if complete and numerical:
            out.update(measured)
        elif not numerical:
            out['complete'] = False
            out['reason'] = 'derived_metric_nonfinite'
    else:
        out.update({'conditional_' + k: np.nan for k in METRIC_FIELDS})
        out['conditional_identity_ok'] = False
    if out['complete'] and (not out['identity_ok']):
        raise ArithmeticError('complete metric identity failed')
    return out

def classify_original(endpoint_rows):
    if len(endpoint_rows) != 2:
        raise ValueError('both endpoint rows required')
    complete = all((row['complete'] and row['identity_ok'] for row in endpoint_rows))
    e = max((row['rmse'] for row in endpoint_rows)) if complete else np.nan
    missing = any(row['missing_series'] for row in endpoint_rows)
    invalid = any(row['invalid_series'] for row in endpoint_rows)
    invalid = invalid or (not complete and not missing)
    return dict(
        complete=bool(complete), e=float(e),
        pass_flag=bool(e <= RMSE_LIMIT) if complete else None,
        margin=RMSE_LIMIT - e if complete else np.nan,
        missing=missing, invalid=invalid,
        reason='' if complete else 'required_reference_scale_or_series_invalid_or_missing',
    )

def group_decision(originals):
    """Fixed30 finite-data classification, no t interval or p value."""
    if len({(r['master_seed'], r['outer_id']) for r in originals}) != len(originals):
        raise ValueError('duplicate original IDs')
    if any((r['master_seed'] not in (0, 1, 2) or r['outer_id'] not in range(10) for r in originals)):
        raise ValueError('out-of-range original ID')
    complete = [r for r in originals if r['complete']]
    passed = sum((r['pass_flag'] is True for r in complete))
    all_complete = len(originals) == 30 and len(complete) == 30
    status = 'inconclusive'
    if all_complete:
        status = 'supported' if passed >= PASS_REQUIRED else 'not_supported'
    return dict(
        planned_N=30, observed_N=len(originals), complete_N=len(complete),
        pass_count=passed, fail_count=len(complete) - passed,
        unknown_count=30 - len(complete),
        invalid_N=sum(bool(row['invalid']) for row in originals),
        missing_N=30 - len(originals) + sum(bool(row['missing']) for row in originals),
        required_pass_count=PASS_REQUIRED, status=status,
        pass_fraction=passed / 30 if all_complete else np.nan,
        conditional_pass_fraction=passed / len(complete) if complete else np.nan,
    )

def joint_status(statuses):
    values = list(statuses)
    return 'not_supported' if 'not_supported' in values else 'supported' if values == ['supported', 'supported'] else 'inconclusive'
