"""RQ-M12: map saved probabilities onto a fixed finite conditional CDF.

No RNG, enumeration or accepted-data loading occurs here. Classifications of
support points are exact; arithmetic identity tolerances never change pass flags.
"""
import numpy as np

SERIES = 2400
ENUM_SIZE = 3125
ENDS = ('lower', 'upper')
LIMIT = .10
NUMERIC_RTOL, NUMERIC_ATOL = 2e-10, 2e-12
IDENTITY_RTOL = IDENTITY_ATOL = 2e-12
UNKNOWN_MOVEMENT = -2


def validate_distribution(support, frequency, cumulative):
    support = np.asarray(support)
    frequency = np.asarray(frequency)
    cumulative = np.asarray(cumulative)
    if (support.ndim != 1 or not support.size or support.dtype != np.float64
            or frequency.shape != support.shape or cumulative.shape != support.shape
            or frequency.dtype != np.int64 or cumulative.dtype != np.int64):
        raise ValueError('fixed CDF shape/dtype differs')
    if (not np.isfinite(support).all() or np.any(support[1:] <= support[:-1])
            or np.any(frequency <= 0) or sum(int(value) for value in frequency) != ENUM_SIZE
            or not np.array_equal(cumulative, np.cumsum(frequency, dtype=np.int64))):
        raise ValueError('fixed support/frequency/cumulative differs')


def cumulative_index(cumulative, probability):
    """First cumulative count >= exact binary64 q; no floating ceil product."""
    if isinstance(probability, (bool, np.bool_)) or not np.isscalar(probability):
        raise ValueError('q is not a real scalar')
    q = float(probability)
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('q outside[0,1]')
    if q == 0:
        return 0
    numerator, denominator = q.as_integer_ratio()
    left, right = 0, len(cumulative) - 1
    while left < right:
        middle = (left + right) // 2
        if int(cumulative[middle]) * denominator >= ENUM_SIZE * numerator:
            right = middle
        else:
            left = middle + 1
    return left


def _pair_probabilities(values):
    return (np.isfinite(values).all(axis=-1)
            & (values[..., 0] >= 0)
            & (values[..., 0] < values[..., 1])
            & (values[..., 1] <= 1))


def analyze_sample(reference, comparison):
    """Return all BCa endpoint arrays and independent percentile controls.

    Method order is accepted M11 BCa/percentile. Missing and numerical-invalid
    slots retain explicit reasons; valid tail warnings are never excluded.
    """
    support = np.asarray(reference['support'])
    frequency = np.asarray(reference['frequency'])
    cumulative = np.asarray(reference['cumulative'])
    validate_distribution(support, frequency, cumulative)
    q_star = np.asarray(reference['q'])
    reference_A = np.asarray(reference['E_cdf'])
    ref_valid = np.asarray(reference['valid'])
    q_B = np.asarray(comparison['saved_q'])
    Y = np.asarray(comparison['saved_endpoint'])
    saved_valid = np.asarray(comparison['saved_valid'])
    observed = np.asarray(comparison['observed'])
    if (q_star.shape != (2, 2) or reference_A.shape != (2, 2)
            or ref_valid.shape != (2,) or ref_valid.dtype != bool
            or q_B.shape != (SERIES, 2, 2) or Y.shape != q_B.shape
            or saved_valid.shape != (SERIES, 2) or saved_valid.dtype != bool
            or observed.shape != (SERIES,) or observed.dtype != bool):
        raise ValueError('accepted method/endpoint/series shape differs')
    scale = float(reference['se'])
    scale_valid = np.isfinite(scale) and scale > 0
    source_delta = np.asarray(comparison['delta'])
    source_error_valid = np.asarray(comparison['error_valid'])
    if source_delta.shape != q_B.shape or source_error_valid.shape != q_B.shape:
        raise ValueError('accepted M11 total-error shape differs')
    labels = ['']
    results = []
    for mi in range(2):
        baseline_indices = np.full(2, -1, dtype=np.int32)
        baseline = np.full(2, np.nan)
        if ref_valid[mi]:
            if not _pair_probabilities(q_star[mi]):
                raise ValueError('valid reference probability is corrupt')
            for ei in range(2):
                atom = cumulative_index(cumulative, q_star[mi, ei])
                baseline_indices[ei] = atom
                baseline[ei] = support[atom]
            if not np.array_equal(baseline, reference_A[mi]):
                raise ValueError('Q(q*) differs from accepted M11 reference')
            if 'atom_index' in reference and not np.array_equal(baseline_indices, reference['atom_index'][mi]):
                raise ValueError('Q(q*) support index differs from M11')
        result = dict(
            q_B=q_B[:, mi].copy(), q_star=q_star[mi].copy(),
            A=baseline, A_index=baseline_indices,
            C=np.full((SERIES, 2), np.nan), Y=Y[:, mi].copy(),
            selected_index=np.full((SERIES, 2), -1, dtype=np.int32),
            valid=np.zeros((SERIES, 2), bool),
            reason_code=np.zeros((SERIES, 2), np.uint8),
            movement_code=np.full((SERIES, 2), UNKNOWN_MOVEMENT, dtype=np.int8),
        )
        pair_valid = _pair_probabilities(q_B[:, mi])
        endpoint_pair = (np.isfinite(Y[:, mi]).all(axis=1)
                         & (Y[:, mi, 0] <= Y[:, mi, 1]))
        cache = {}
        for si in range(SERIES):
            for ei in range(2):
                reason = ''
                if not observed[si]:
                    reason = 'not_observed'
                elif not ref_valid[mi]:
                    reason = 'M11_reference:' + str(reference['reason'][mi])
                elif not scale_valid:
                    reason = 'sample_SE_invalid'
                elif not saved_valid[si, mi]:
                    source = str(comparison['saved_reason'][si, mi])
                    reason = 'M10_saved_invalid:' + (source or 'unspecified')
                elif not pair_valid[si]:
                    reason = 'saved_probability_nonfinite_outside_or_unordered'
                elif not endpoint_pair[si]:
                    reason = 'saved_endpoint_nonfinite_or_unordered'
                elif not source_error_valid[si, mi, ei] or not np.isfinite(source_delta[si, mi, ei]):
                    reason = 'M11_total_error_invalid'
                if reason:
                    if reason not in labels:
                        labels.append(reason)
                    result['reason_code'][si, ei] = labels.index(reason)
                    continue
                probability = float(q_B[si, mi, ei])
                if probability not in cache:
                    cache[probability] = cumulative_index(cumulative, probability)
                atom = cache[probability]
                result['selected_index'][si, ei] = atom
                result['C'][si, ei] = support[atom]
                result['valid'][si, ei] = True
                result['movement_code'][si, ei] = np.sign(atom - baseline_indices[ei])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result['g'] = (result['C'] - baseline) / scale
            result['h'] = (result['Y'] - result['C']) / scale
            result['d'] = (result['Y'] - baseline) / scale
            result['q_delta'] = result['q_B'] - result['q_star']
        for ei in range(2):
            valid = result['valid'][:, ei]
            finite = np.isfinite(np.column_stack([result[field][:, ei] for field in ('g', 'h', 'd', 'q_delta')])).all(axis=1)
            numerical_invalid = valid & ~finite
            if numerical_invalid.any():
                reason = 'derived_movement_nonfinite'
                if reason not in labels:
                    labels.append(reason)
                result['reason_code'][numerical_invalid, ei] = labels.index(reason)
                result['valid'][numerical_invalid, ei] = False
                result['movement_code'][numerical_invalid, ei] = UNKNOWN_MOVEMENT
                result['selected_index'][numerical_invalid, ei] = -1
            valid = result['valid'][:, ei]
            if not np.allclose(result['d'][valid, ei], source_delta[valid, mi, ei],
                               rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL):
                raise ArithmeticError('direct total error differs from accepted M11 delta')
        for field in ('C', 'g', 'h', 'd', 'q_delta'):
            result[field][~result['valid']] = np.nan
        result['q_changed'] = result['valid'] & (result['q_B'] != result['q_star'])
        result['q_changed_same'] = result['q_changed'] & (result['movement_code'] == 0)
        result['support_step'] = result['selected_index'] - result['A_index']
        result['support_step'][~result['valid']] = np.iinfo(np.int32).min
        result['abs_g'] = np.abs(result['g'])
        result['total_error_residual'] = result['d'] - source_delta[:, mi]
        result['endpoint_identity_residual'] = result['d'] - (result['g'] + result['h'])
        results.append(result)
    if len(labels) > 256:
        raise ValueError('reason dictionary overflow')
    control = results[1]
    if np.any(control['g'][control['valid']] != 0):
        raise ArithmeticError('fixed-probability percentile movement is nonzero')
    primary = results[0]
    return {
        **primary, 'se': scale,
        'observed': observed.copy(),
        'source_warning': comparison['saved_warning'][:, 0].copy(),
        'source_tail_count': np.column_stack([comparison['saved_tau_low'][:, 0], comparison['saved_tau_high'][:, 0]]),
        'reason_labels': np.asarray(labels, dtype='U200'),
        'percentile_q_B': control['q_B'], 'percentile_q_star': control['q_star'],
        'percentile_A': control['A'], 'percentile_C': control['C'],
        'percentile_selected_index': control['selected_index'],
        'percentile_reference_index': control['A_index'],
        'percentile_g': control['g'], 'percentile_valid': control['valid'],
        'percentile_reason_code': control['reason_code'],
        'support': support.copy(), 'frequency': frequency.copy(),
        'cumulative': cumulative.copy(), 'cdf_probability': cumulative / ENUM_SIZE,
    }


def endpoint_metrics(data, endpoint):
    ei = endpoint
    valid = data['valid'][:, ei]
    observed = data['observed']
    codes = data['movement_code'][:, ei]
    n = int(valid.sum())
    complete = n == SERIES
    same = int((valid & (codes == 0)).sum())
    down = int((valid & (codes == -1)).sum())
    up = int((valid & (codes == 1)).sum())
    moved = down + up
    q_changed = int((valid & data['q_changed'][:, ei]).sum())
    q_changed_same = int((valid & data['q_changed_same'][:, ei]).sum())
    output = dict(
        planned_series=SERIES, observed_series=int(observed.sum()), valid_series=n,
        invalid_series=int((observed & ~valid).sum()), missing_series=int((~observed).sum()),
        complete=complete, reason='' if complete else 'required_input_invalid_or_missing',
        same_count=same, down_count=down, up_count=up, moved_count=moved,
        q_changed_count=q_changed, q_changed_same_count=q_changed_same,
        source_warning_series=int((observed & data['source_warning']).sum()),
        percentile_valid_series=int(data['percentile_valid'][:, ei].sum()),
        percentile_complete=bool(data['percentile_valid'][:, ei].all()),
        percentile_g_nonzero_count=int((data['percentile_g'][data['percentile_valid'][:, ei], ei] != 0).sum()),
    )
    fields = (
        'same_rate', 'down_rate', 'up_rate', 'moved_rate', 'q_changed_rate',
        'q_changed_same_rate', 'same_given_q_changed_rate', 'mean_g', 'rms_g',
        'mean_abs_g', 'abs_g_p95', 'abs_g_max', 'moved_abs_g_mean',
        'mean_q_delta', 'q_delta_min', 'q_delta_max', 'mean_abs_q_delta',
        'abs_q_delta_p95', 'abs_q_delta_max', 'mean_h', 'rms_h', 'mean_d', 'rms_d',
        'mean_g2', 'mean_h2', 'mean_d2', 'cross_term',
        'endpoint_identity_max_abs', 'endpoint_identity_max_normalized',
        'moment_identity_residual', 'moment_identity_scale', 'moment_identity_normalized',
        'M11_delta_max_abs',
    )
    output.update({key: np.nan for key in fields})
    output['identity_ok'] = False
    if not n:
        output.update({'conditional_' + key: np.nan for key in fields})
        output['conditional_identity_ok'] = False
        return output
    g, h, d, q_delta = [data[key][valid, ei] for key in ('g', 'h', 'd', 'q_delta')]
    abs_g, abs_q = np.abs(g), np.abs(q_delta)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        g2, h2, d2 = [float(np.mean(array**2)) for array in (g, h, d)]
        cross = float(2 * np.mean(g * h))
        moment_residual = d2 - (g2 + h2 + cross)
        moment_scale = max(abs(d2), abs(g2), abs(h2), abs(cross), abs(g2 + h2 + cross))
        point_residual = d - (g + h)
        point_scale = np.maximum.reduce([np.abs(g), np.abs(h), np.abs(d), np.ones(n)])
        point_ok = np.abs(point_residual) <= IDENTITY_ATOL + IDENTITY_RTOL * point_scale
        moment_ok = abs(moment_residual) <= IDENTITY_ATOL + IDENTITY_RTOL * moment_scale
    moved_mask = codes[valid] != 0
    values = dict(
        same_rate=same/n, down_rate=down/n, up_rate=up/n, moved_rate=moved/n,
        q_changed_rate=q_changed/n, q_changed_same_rate=q_changed_same/n,
        same_given_q_changed_rate=q_changed_same/q_changed if q_changed else np.nan,
        mean_g=float(g.mean()), rms_g=float(np.sqrt(g2)), mean_abs_g=float(abs_g.mean()),
        abs_g_p95=float(np.quantile(abs_g, .95, method='linear')), abs_g_max=float(abs_g.max()),
        moved_abs_g_mean=float(abs_g[moved_mask].mean()) if moved else np.nan,
        mean_q_delta=float(q_delta.mean()), q_delta_min=float(q_delta.min()),
        q_delta_max=float(q_delta.max()), mean_abs_q_delta=float(abs_q.mean()),
        abs_q_delta_p95=float(np.quantile(abs_q, .95, method='linear')), abs_q_delta_max=float(abs_q.max()),
        mean_h=float(h.mean()), rms_h=float(np.sqrt(h2)), mean_d=float(d.mean()), rms_d=float(np.sqrt(d2)),
        mean_g2=g2, mean_h2=h2, mean_d2=d2, cross_term=cross,
        endpoint_identity_max_abs=float(np.max(np.abs(point_residual))),
        endpoint_identity_max_normalized=float(np.max(np.abs(point_residual)/point_scale)),
        moment_identity_residual=moment_residual, moment_identity_scale=moment_scale,
        moment_identity_normalized=moment_residual/max(1., moment_scale),
        M11_delta_max_abs=float(np.max(np.abs(data['total_error_residual'][valid, ei]))),
    )
    needed = [value for key, value in values.items() if key not in ('same_given_q_changed_rate', 'moved_abs_g_mean')]
    numerical = bool(np.isfinite(needed).all())
    identity_ok = bool(numerical and point_ok.all() and moment_ok)
    if numerical and not identity_ok:
        raise ArithmeticError('probability-movement decomposition identity failed')
    output.update({'conditional_' + key: value for key, value in values.items()})
    output['conditional_identity_ok'] = identity_ok
    if complete and numerical:
        output.update(values, identity_ok=identity_ok)
    elif not numerical:
        output.update(complete=False, reason='derived_metric_nonfinite')
    return output


def classify_original(endpoints):
    if len(endpoints) != 2:
        raise ValueError('both endpoints required')
    complete = all(row['complete'] and row['identity_ok'] for row in endpoints)
    missing = any(row['missing_series'] for row in endpoints)
    invalid = (any(row['invalid_series'] for row in endpoints)
               or any(row.get('reason') == 'derived_metric_nonfinite' for row in endpoints)
               or (not complete and not missing))
    u = max(row['rms_g'] for row in endpoints) if complete else np.nan
    return dict(complete=complete, u=u, margin=LIMIT-u if complete else np.nan,
                pass_flag=bool(u <= LIMIT) if complete else None,
                invalid=bool(invalid), missing=bool(missing),
                reason='' if complete else 'required_reference_scale_or_series_invalid_or_missing')


def group_decision(rows):
    identities = {(row['master_seed'], row['outer_id']) for row in rows}
    if len(identities) != len(rows) or any(s not in (0,1,2) or o not in range(10) for s,o in identities):
        raise ValueError('original IDs duplicate/outside fixed group')
    complete = [row for row in rows if row['complete']]
    passed = sum(row['pass_flag'] is True for row in complete)
    all_complete = len(rows) == len(complete) == 30
    status = 'inconclusive'
    if all_complete:
        status = 'supported' if passed >= 27 else 'not_supported'
    return dict(planned_N=30, observed_N=len(rows), complete_N=len(complete),
        pass_count=passed, fail_count=len(complete)-passed, unknown_N=30-len(complete),
        invalid_N=sum(bool(row['invalid']) for row in rows),
        missing_N=30-len(rows)+sum(bool(row['missing']) for row in rows),
        required_pass_count=27, status=status,
        pass_fraction=passed/30 if all_complete else np.nan,
        conditional_pass_fraction=passed/len(complete) if complete else np.nan)


def joint_status(statuses):
    values = list(statuses)
    if 'not_supported' in values:
        return 'not_supported'
    return 'supported' if values == ['supported', 'supported'] else 'inconclusive'
