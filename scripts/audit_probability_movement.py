"""Independent saved-only RQ-M12 audit: integer ranks on sorted M11 means.

No primary M12 computation is imported. The accepted independent M11 finite-B
interval primitive is reused only to connect the existing 36 archived index
streams to M10 endpoints and probabilities; no M11 analysis is rerun.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_enumerated_reference import reference_intervals
DISTRIBUTIONS = ('gamma', 'lognormal')
ENDPOINTS = ('lower', 'upper')
ACCEPTED_HEAD = '0deb14fa313dc27380afbb43e6b132386aa40206'
ENUM_SIZE = 3125
SERIES = 2400
NUMERIC_RTOL, NUMERIC_ATOL = 2e-10, 2e-12
IDENTITY_RTOL, IDENTITY_ATOL = 2e-12, 2e-12


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_hash(sample):
    return hashlib.sha256(np.asarray(sample, dtype='<f8').tobytes(order='C')).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def scalar(saved, field):
    value = np.asarray(saved[field])
    if value.shape != ():
        raise ValueError(f'{field}: expected scalar metadata')
    return value.item()


def select_rank(sorted_means, q):
    """Independent 1-based exact binary64 rank, never a float ceil operation."""
    values = np.asarray(sorted_means)
    if values.shape != (ENUM_SIZE,) or values.dtype != np.float64:
        raise ValueError('expected 3125 sorted binary64 means')
    if not np.isfinite(values).all() or np.any(values[1:] < values[:-1]):
        raise ValueError('finite means must be nondecreasing')
    if isinstance(q, (bool, np.bool_)) or not np.isscalar(q):
        raise ValueError('q must be a scalar probability')
    q = float(q)
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('q outside [0,1]')
    numerator, denominator = q.as_integer_ratio()
    rank = max(1, (ENUM_SIZE*numerator+denominator-1)//denominator)
    return float(values[rank-1]), int(rank)


def verify_distribution(reference):
    means = reference['means']
    if reference['support'].dtype!=np.float64 or reference['cumulative'].dtype!=np.int64:
        raise ValueError('M11 support/cumulative storage differs')
    for field in ('q','E_cdf'):
        if reference[field].shape!=(2,2) or reference[field].dtype!=np.float64:
            raise ValueError(f'M11 {field} shape/dtype differs')
    if reference['atom_index'].shape!=(2,2) or reference['atom_index'].dtype!=np.int64:
        raise ValueError('M11 reference atom storage differs')
    if reference['valid'].shape!=(2,) or reference['valid'].dtype!=bool:
        raise ValueError('M11 reference mask storage differs')
    if means.shape != (ENUM_SIZE,) or means.dtype != np.float64 or not np.isfinite(means).all():
        raise ValueError('M11 ordered mean storage differs')
    sorted_means = np.sort(means)
    support, frequency = np.unique(sorted_means, return_counts=True)
    if not np.array_equal(support, reference['support']):
        raise ValueError('M11 support differs from sorted stored means')
    if not np.array_equal(frequency, reference['frequency']):
        raise ValueError('M11 frequencies differ from sorted stored means')
    if reference['frequency'].dtype != np.int64 or np.any(reference['frequency'] <= 0):
        raise ValueError('M11 frequency storage invalid')
    if not np.array_equal(np.cumsum(frequency, dtype=np.int64), reference['cumulative']):
        raise ValueError('M11 cumulative integer frequencies differ')
    for mi in range(2):
        if not reference['valid'][mi]:
            continue
        for ei in range(2):
            selected, _ = select_rank(sorted_means, reference['q'][mi, ei])
            if selected != reference['E_cdf'][mi, ei]:
                raise ValueError('M11 reference differs from exact sorted-mean rank')
            if selected != reference['support'][reference['atom_index'][mi, ei]]:
                raise ValueError('M11 selected support point differs')
    return sorted_means


def identity_check(residual, scale):
    return bool(np.isfinite(residual) and np.isfinite(scale) and
                abs(residual) <= IDENTITY_ATOL+IDENTITY_RTOL*scale)


def population_decision(rows):
    identities = {(row['master_seed'], row['outer_id']) for row in rows}
    if len(identities) != len(rows) or any(seed not in (0,1,2) or outer not in range(10) for seed,outer in identities):
        raise ValueError('duplicate/out-of-range original ID')
    complete = [row for row in rows if row['complete']]
    passes = sum(row['pass_flag'] is True for row in complete)
    if len(rows) != 30 or len(complete) != 30:
        status = 'inconclusive'
    else:
        status = 'supported' if passes >= 27 else 'not_supported'
    return dict(planned_N=30, observed_N=len(rows), complete_N=len(complete),
                pass_count=passes, fail_count=len(complete)-passes,
                unknown_N=30-len(complete), invalid_N=sum(bool(row['invalid']) for row in rows),
                missing_N=30-len(rows)+sum(bool(row['missing']) for row in rows),
                required_pass_count=27,status=status,pass_fraction=passes/30 if len(rows)==len(complete)==30 else np.nan,
                conditional_pass_fraction=passes/len(complete) if complete else np.nan)


def joint_status(statuses):
    if 'not_supported' in statuses:
        return 'not_supported'
    return 'supported' if len(statuses)==2 and all(item=='supported' for item in statuses) else 'inconclusive'


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_bytes())


def assert_value(expected, actual, location, exact=False):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual)!=set(expected):
            raise ValueError(f'{location}: object fields differ')
        for field,value in expected.items():
            assert_value(value,actual[field],location+'/'+field,exact=exact)
        return
    if isinstance(expected,(tuple,list)):
        if not isinstance(actual,(tuple,list)) or len(expected)!=len(actual):
            raise ValueError(f'{location}: list shape differs')
        for index,(first,second) in enumerate(zip(expected,actual)):
            assert_value(first,second,f'{location}/{index}',exact=exact)
        return
    if isinstance(expected,(bool,np.bool_)):
        matched = actual is bool(expected) or str(actual)==str(bool(expected))
    elif expected is None or isinstance(expected,(float,np.floating)) and np.isnan(expected):
        try:
            matched = np.isnan(float(actual))
        except (TypeError,ValueError):
            matched = actual is None or actual in ('','NA')
    elif isinstance(expected,(int,np.integer)):
        try:
            matched = int(actual)==int(expected) and float(actual)==int(expected)
        except (TypeError,ValueError,OverflowError):
            matched = False
    elif isinstance(expected,(float,np.floating)):
        try:
            number = float(actual)
            matched = number==float(expected) if exact else np.isclose(number,expected,rtol=NUMERIC_RTOL,atol=NUMERIC_ATOL)
        except (TypeError,ValueError):
            matched = False
    else:
        matched = str(expected)==str(actual)
    if not matched:
        raise ValueError(f'{location}: value differs')


def assert_arrays(expected, actual, label, exact_fields=()):
    if set(expected)!=set(actual):
        raise ValueError(f'{label}: field schema differs')
    for field,value in expected.items():
        first,second=np.asarray(value),np.asarray(actual[field])
        if first.shape!=second.shape or second.dtype.hasobject:
            raise ValueError(f'{label}/{field}: shape/object storage differs')
        exact = first.dtype.kind in 'biuUS' or field in exact_fields
        if exact:
            equal=np.array_equal(first,second,equal_nan=True) if first.dtype.kind=='f' else np.array_equal(first,second)
        else:
            equal=np.allclose(first,second,rtol=NUMERIC_RTOL,atol=NUMERIC_ATOL,equal_nan=True)
        if not equal:
            raise ValueError(f'{label}/{field}: values differ')
        if first.dtype.kind in 'bifu' and first.dtype!=second.dtype:
            raise ValueError(f'{label}/{field}: numeric storage dtype differs')
        if first.dtype.kind in 'US' and first.dtype.kind!=second.dtype.kind:
            raise ValueError(f'{label}/{field}: text storage differs')


def verify_receipt(root, output):
    root,output=Path(root),Path(output)
    manifest=read_json(output/'manifest.json')
    receipt=read_json(output/'input_receipt.json')
    if manifest['question_id']!='RQ-M12' or receipt['question_id']!='RQ-M12':
        raise ValueError('analysis question differs')
    if manifest['accepted_head']!=ACCEPTED_HEAD or receipt['accepted_head']!=ACCEPTED_HEAD:
        raise ValueError('accepted M11 head differs')
    if receipt['inputs']!=manifest['source_inputs'] or receipt['originals']!=manifest['originals']:
        raise ValueError('input receipt differs from frozen manifest')
    if receipt['manifest_sha256']!=digest(output/'manifest.json'):
        raise ValueError('input receipt manifest hash differs')
    commit=receipt['preanalysis_commit']
    if not isinstance(commit,str) or len(commit)!=40 or any(character not in '0123456789abcdef' for character in commit):
        raise ValueError('preanalysis commit is not a full SHA')
    hashes={}
    for relative,expected in receipt['inputs'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe accepted input path')
        content=(root/path).read_bytes()
        sha=hashlib.sha256(content).hexdigest()
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if expected!={'sha256':sha,'git_blob':blob}:
            raise ValueError(f'accepted input bytes differ: {relative}')
        hashes[relative]=sha
    minimum={'evidence/rq_m11/input_receipt.json','evidence/rq_m11/original_metrics.csv',
             'evidence/rq_m10/originals/receipt.json'}
    identities=[]
    for dist in DISTRIBUTIONS:
        for seed in range(3):
            stem=f'{dist}_seed{seed}'
            minimum.update(f'evidence/rq_m10/{folder}/{stem}.npz' for folder in ('panels','originals'))
            minimum.update(f'evidence/rq_m10/streams/{stem}_outer0_repeat0_stream{k}.npz' for k in (0,29,30,59,60,119))
            for outer in range(10):
                identities.append((dist,seed,outer))
                minimum.update(f'evidence/rq_m11/{folder}/{stem}_outer{outer}.npz' for folder in ('references','comparisons'))
    if not minimum.issubset(receipt['inputs']):
        raise ValueError('required accepted input categories/IDs missing')
    if [(record['distribution'],record['master_seed'],record['outer_id']) for record in receipt['originals']]!=identities:
        raise ValueError('fixed sixty original IDs/order differ')
    if receipt['originals']!=read_json(root/'evidence/rq_m11/input_receipt.json')['originals']:
        raise ValueError('M11 original receipt correspondence differs')
    for field,expected in dict(sample_size=5,B=4999,alpha=.05,originals_per_population=30,
                               paired_streams_per_original=2400,total_distinct_saved_series=144000,
                               BCa_endpoint_observations=288000).items():
        if manifest[field]!=expected:
            raise ValueError(f'fixed manifest design differs: {field}')
    return manifest,receipt,hashes


def verify_archived_stream(path, panel, distribution, seed, stream):
    """Only these36 finite-B generation bridges are reconstructible from indices."""
    saved=load_npz(path)
    indices,means=saved['indices'],saved['means']
    if indices.shape!=(4999,5) or indices.dtype!=np.uint8 or np.any(indices>=5):
        raise ValueError('archived indices shape/dtype/range differ')
    if means.shape!=(4999,) or means.dtype!=np.float64:
        raise ValueError('archived means shape/dtype differ')
    sample=panel['samples'][0]
    if not np.array_equal(np.mean(sample[indices],axis=1),means):
        raise ValueError('archived means differ from saved indices')
    metadata=dict(distribution=distribution,master_seed=seed,outer_id=0,budget_repeat_id=0,
                  stream_id=stream,n=5,B=4999,R=20,streams_per_budget=120,alpha=.05,
                  namespace=0x4D3130,native_indices_dtype='int64',stored_indices_dtype='uint8')
    for field,expected in metadata.items():
        if scalar(saved,field)!=expected:
            raise ValueError(f'archived stream metadata differs: {field}')
    archived_hash=scalar(saved,'sample_sha256')
    if isinstance(archived_hash,bytes):
        archived_hash=archived_hash.decode()
    if archived_hash!=sample_hash(sample):
        raise ValueError('archived stream sample hash differs')
    reconstructed=reference_intervals(sample,means)
    for mi,method in enumerate(('t','percentile','bca')):
        for field,expected in reconstructed[method].items():
            actual=(str(panel['reason_labels'][panel['reason_codes'][0,0,mi,stream]]) if field=='reason'
                    else panel[field][0,0,mi,stream].item())
            assert_value(expected,actual,f'archived/{distribution}/{seed}/{stream}/{method}/{field}')


def validate_source_original(root, identity, panel):
    """Join M11 saved arrays directly to their accepted M10 positions and bytes."""
    dist,seed,outer=identity
    stem=f'{dist}_seed{seed}_outer{outer}.npz'
    base=Path(root)/'evidence/rq_m11'
    reference=load_npz(base/'references'/stem)
    comparison=load_npz(base/'comparisons'/stem)
    sample=panel['samples'][outer]
    digest_value=sample_hash(sample)
    if sample.shape!=(5,) or sample.dtype!=np.float64 or not np.isfinite(sample).all():
        raise ValueError('M10 original observation storage differs')
    if panel['sample_sha256'][outer].decode()!=digest_value:
        raise ValueError('M10 sample hash differs')
    for saved in (reference,comparison):
        for field,expected in dict(distribution=dist,master_seed=seed,outer_id=outer,
                                   sample_sha256=digest_value,source_question_id='RQ-M10',
                                   question_id='RQ-M11',n=5,B=4999,R=20,K=120).items():
            if scalar(saved,field)!=expected:
                raise ValueError(f'M11 original metadata differs: {field}')
        if tuple(saved['methods'].astype(str))!=('bca','percentile') or tuple(saved['endpoints'].astype(str))!=ENDPOINTS:
            raise ValueError('M11 saved method/endpoint axes differ')
    if np.all(sample==sample[0]):
        se=np.nan
    else:
        with np.errstate(over='ignore',invalid='ignore'):
            se=float(np.std(sample,ddof=1)/np.sqrt(5))
    assert_value(se,scalar(reference,'se'),'M11/sample_SE')
    if not np.array_equal(comparison['budget_repeat_id'],np.repeat(np.arange(20),120)) or not np.array_equal(comparison['stream_id'],np.tile(np.arange(120),20)):
        raise ValueError('M11 series r*120+k ordering differs')
    if not np.array_equal(comparison['observed'],panel['observed_streams'][outer].reshape(-1)):
        raise ValueError('M11 observed mask differs from M10')
    expected_endpoint=np.stack([
        np.column_stack([panel[endpoint][outer,:,mi,:].reshape(-1) for endpoint in ENDPOINTS])
        for mi in (2,1)],axis=1)
    expected_q=np.stack([
        np.column_stack([panel[field][outer,:,mi,:].reshape(-1) for field in ('q_low','q_high')])
        for mi in (2,1)],axis=1)
    for field,expected in (('saved_endpoint',expected_endpoint),('saved_q',expected_q)):
        if comparison[field].shape!=(2400,2,2) or comparison[field].dtype!=np.float64 or not np.array_equal(comparison[field],expected,equal_nan=True):
            raise ValueError(f'M11 {field} differs from M10 saved values')
    aliases=dict(saved_valid='valid',saved_warning='tail_warning',saved_z0='z0',saved_a='a',
                 less_count='less_count',equal_count='equal_count',saved_tau_low='tau_low',saved_tau_high='tau_high')
    for destination,field in aliases.items():
        expected=np.column_stack([panel[field][outer,:,mi,:].reshape(-1) for mi in (2,1)])
        actual=comparison[destination]
        equal=np.array_equal(actual,expected,equal_nan=True) if actual.dtype.kind=='f' else np.array_equal(actual,expected)
        if actual.shape!=(2400,2) or actual.dtype!=expected.dtype or not equal:
            raise ValueError(f'M11 {destination} differs from M10 saved values')
    expected_reason=panel['reason_labels'][panel['reason_codes'][outer]]
    expected_reason=np.column_stack([expected_reason[:,mi,:].reshape(-1) for mi in (2,1)])
    if not np.array_equal(comparison['saved_reason'],expected_reason):
        raise ValueError('M11 saved invalid reasons differ from M10')
    for field in ('delta','error_valid'):
        if comparison[field].shape!=(2400,2,2):
            raise ValueError(f'M11 {field} shape differs')
    if comparison['observed'].dtype!=bool or comparison['saved_valid'].dtype!=bool or comparison['error_valid'].dtype!=bool:
        raise ValueError('M11 mask storage dtype differs')
    sorted_means=verify_distribution(reference)
    return reference,comparison,sorted_means


def map_original(reference, comparison):
    """Select every value by sorted-mean rank; no cumulative-CDF search."""
    ordered=verify_distribution(reference)
    support=reference['support']
    se=float(reference['se'])
    observed=comparison['observed']
    labels=['']
    methods=[]
    def reason_number(reason):
        if reason not in labels:
            labels.append(reason)
        return labels.index(reason)
    for method in range(2):
        probabilities=comparison['saved_q'][:,method].copy()
        Y=comparison['saved_endpoint'][:,method].copy()
        qstar=reference['q'][method].copy()
        A=np.full(2,np.nan)
        baseline_index=np.full(2,-1,dtype=np.int32)
        if reference['valid'][method]:
            if not np.isfinite(qstar).all() or not 0<=qstar[0]<qstar[1]<=1:
                raise ValueError('valid reference q is corrupt')
            for ei in range(2):
                A[ei],_=select_rank(ordered,qstar[ei])
                baseline_index[ei]=np.searchsorted(support,A[ei])
            if not np.array_equal(A,reference['E_cdf'][method]) or not np.array_equal(baseline_index,reference['atom_index'][method]):
                raise ValueError('reference selected atom differs')
        result=dict(q_B=probabilities,q_star=qstar,A=A,A_index=baseline_index,Y=Y,
                    C=np.full((SERIES,2),np.nan),selected_index=np.full((SERIES,2),-1,dtype=np.int32),
                    valid=np.zeros((SERIES,2),bool),reason_code=np.zeros((SERIES,2),np.uint8),
                    movement_code=np.full((SERIES,2),-2,dtype=np.int8))
        ranks={}
        for si in range(SERIES):
            pair=probabilities[si]
            for ei in range(2):
                reason=''
                if not observed[si]:
                    reason='not_observed'
                elif not reference['valid'][method]:
                    reason='M11_reference:'+str(reference['reason'][method])
                elif not np.isfinite(se) or se<=0:
                    reason='sample_SE_invalid'
                elif not comparison['saved_valid'][si,method]:
                    reason='M10_saved_invalid:'+(str(comparison['saved_reason'][si,method]) or 'unspecified')
                elif not np.isfinite(pair).all() or not 0<=pair[0]<pair[1]<=1:
                    reason='saved_probability_nonfinite_outside_or_unordered'
                elif not np.isfinite(Y[si]).all() or Y[si,0]>Y[si,1]:
                    reason='saved_endpoint_nonfinite_or_unordered'
                elif not comparison['error_valid'][si,method,ei] or not np.isfinite(comparison['delta'][si,method,ei]):
                    reason='M11_total_error_invalid'
                if reason:
                    result['reason_code'][si,ei]=reason_number(reason)
                    continue
                q=float(pair[ei])
                if q not in ranks:
                    numerator,denominator=q.as_integer_ratio()
                    rank=max(1,(3125*numerator+denominator-1)//denominator)
                    value=float(ordered[rank-1])
                    atom=int(np.searchsorted(support,value))
                    ranks[q]=(value,atom)
                selected,atom=ranks[q]
                result['selected_index'][si,ei]=atom
                result['C'][si,ei]=selected
                result['valid'][si,ei]=True
                result['movement_code'][si,ei]=0 if selected==A[ei] else 1 if selected>A[ei] else -1
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            result['g']=(result['C']-A)/se
            result['h']=(Y-result['C'])/se
            result['d']=(Y-A)/se
            result['q_delta']=probabilities-qstar
        for ei in range(2):
            finite=np.isfinite(np.column_stack([result[key][:,ei] for key in ('g','h','d','q_delta')])).all(axis=1)
            numerical_invalid=result['valid'][:,ei]&~finite
            if numerical_invalid.any():
                result['reason_code'][numerical_invalid,ei]=reason_number('derived_movement_nonfinite')
                result['valid'][numerical_invalid,ei]=False
                result['movement_code'][numerical_invalid,ei]=-2
            valid=result['valid'][:,ei]
            if not np.allclose(result['d'][valid,ei],comparison['delta'][valid,method,ei],rtol=NUMERIC_RTOL,atol=NUMERIC_ATOL):
                raise ValueError('direct total error differs from accepted M11 delta')
        for key in ('C','g','h','d','q_delta'):
            result[key][~result['valid']]=np.nan
        result['q_changed']=result['valid']&(probabilities!=qstar)
        result['q_changed_same']=result['q_changed']&(result['movement_code']==0)
        result['support_step']=result['selected_index']-baseline_index
        result['support_step'][~result['valid']]=np.iinfo(np.int32).min
        result['abs_g']=np.abs(result['g'])
        result['total_error_residual']=result['d']-comparison['delta'][:,method]
        result['endpoint_identity_residual']=result['d']-(result['g']+result['h'])
        methods.append(result)
    control=methods[1]
    if np.any(control['g'][control['valid']]!=0):
        raise ValueError('fixed percentile probability selects a different point')
    return {**methods[0], 'se':se,'observed':observed.copy(),
            'source_warning':comparison['saved_warning'][:,0].copy(),
            'source_tail_count':np.column_stack([comparison['saved_tau_low'][:,0],comparison['saved_tau_high'][:,0]]),
            'reason_labels':np.asarray(labels,dtype='U200'),
            'percentile_q_B':control['q_B'],'percentile_q_star':control['q_star'],
            'percentile_A':control['A'],'percentile_C':control['C'],
            'percentile_selected_index':control['selected_index'],'percentile_reference_index':control['A_index'],
            'percentile_g':control['g'],'percentile_valid':control['valid'],'percentile_reason_code':control['reason_code'],
            'support':support.copy(),'frequency':reference['frequency'].copy(),
            'cumulative':reference['cumulative'].copy(),'cdf_probability':reference['cumulative']/3125}


STAT_FIELDS=('same_rate','down_rate','up_rate','moved_rate','q_changed_rate','q_changed_same_rate',
             'same_given_q_changed_rate','mean_g','rms_g','mean_abs_g','abs_g_p95','abs_g_max',
             'moved_abs_g_mean','mean_q_delta','q_delta_min','q_delta_max','mean_abs_q_delta',
             'abs_q_delta_p95','abs_q_delta_max','mean_h','rms_h','mean_d','rms_d','mean_g2','mean_h2',
             'mean_d2','cross_term','endpoint_identity_max_abs','endpoint_identity_max_normalized',
             'moment_identity_residual','moment_identity_scale','moment_identity_normalized','M11_delta_max_abs')


def endpoint_metrics(data, ei):
    valid=data['valid'][:,ei]
    n=int(valid.sum())
    code=data['movement_code'][:,ei]
    same=int(np.count_nonzero(valid&(code==0)))
    down=int(np.count_nonzero(valid&(code==-1)))
    up=int(np.count_nonzero(valid&(code==1)))
    changed=int(np.count_nonzero(valid&data['q_changed'][:,ei]))
    changed_same=int(np.count_nonzero(valid&data['q_changed_same'][:,ei]))
    observed=data['observed']
    pct_valid=data['percentile_valid'][:,ei]
    output=dict(planned_series=2400,observed_series=int(observed.sum()),valid_series=n,
                invalid_series=int(np.count_nonzero(observed&~valid)),missing_series=int(np.count_nonzero(~observed)),
                complete=n==2400,reason='' if n==2400 else 'required_input_invalid_or_missing',
                same_count=same,down_count=down,up_count=up,moved_count=down+up,
                q_changed_count=changed,q_changed_same_count=changed_same,
                source_warning_series=int(np.count_nonzero(observed&data['source_warning'])),
                percentile_valid_series=int(pct_valid.sum()),percentile_complete=bool(pct_valid.all()),
                percentile_g_nonzero_count=int(np.count_nonzero(data['percentile_g'][pct_valid,ei]!=0)),identity_ok=False)
    output.update({key:np.nan for key in STAT_FIELDS})
    if not n:
        output.update({'conditional_'+key:np.nan for key in STAT_FIELDS})
        output['conditional_identity_ok']=False
        return output
    g,h,d,qdiff=[data[key][valid,ei] for key in ('g','h','d','q_delta')]
    absolute=np.abs(g)
    qabsolute=np.abs(qdiff)
    moved=code[valid]!=0
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        g2,h2,d2=[float(np.mean(value**2)) for value in (g,h,d)]
        cross=float(2*np.mean(g*h))
        residual=d2-(g2+h2+cross)
        scale=max(abs(d2),abs(g2),abs(h2),abs(cross),abs(g2+h2+cross))
        point=d-(g+h)
        point_scale=np.maximum.reduce([np.ones(n),abs(g),abs(h),abs(d)])
    values=dict(same_rate=same/n,down_rate=down/n,up_rate=up/n,moved_rate=(down+up)/n,
                q_changed_rate=changed/n,q_changed_same_rate=changed_same/n,
                same_given_q_changed_rate=changed_same/changed if changed else np.nan,
                mean_g=float(np.mean(g)),rms_g=float(np.sqrt(g2)),mean_abs_g=float(absolute.mean()),
                abs_g_p95=float(np.quantile(absolute,.95,method='linear')),abs_g_max=float(absolute.max()),
                moved_abs_g_mean=float(absolute[moved].mean()) if moved.any() else np.nan,
                mean_q_delta=float(qdiff.mean()),q_delta_min=float(qdiff.min()),q_delta_max=float(qdiff.max()),
                mean_abs_q_delta=float(qabsolute.mean()),abs_q_delta_p95=float(np.quantile(qabsolute,.95,method='linear')),
                abs_q_delta_max=float(qabsolute.max()),mean_h=float(h.mean()),rms_h=float(np.sqrt(h2)),
                mean_d=float(d.mean()),rms_d=float(np.sqrt(d2)),mean_g2=g2,mean_h2=h2,mean_d2=d2,
                cross_term=cross,endpoint_identity_max_abs=float(np.max(abs(point))),
                endpoint_identity_max_normalized=float(np.max(abs(point)/point_scale)),
                moment_identity_residual=residual,moment_identity_scale=scale,
                moment_identity_normalized=residual/max(1.,scale),
                M11_delta_max_abs=float(np.max(abs(data['total_error_residual'][valid,ei]))))
    numerical=np.isfinite([value for key,value in values.items() if key not in ('same_given_q_changed_rate','moved_abs_g_mean')]).all()
    identities=bool(numerical and np.all(abs(point)<=IDENTITY_ATOL+IDENTITY_RTOL*point_scale) and identity_check(residual,scale))
    if numerical and not identities:
        raise ValueError('movement decomposition identity fails')
    output.update({'conditional_'+key:value for key,value in values.items()})
    output['conditional_identity_ok']=identities
    if output['complete'] and numerical:
        output.update(values,identity_ok=identities)
    elif not numerical:
        output.update(complete=False,reason='derived_metric_nonfinite')
    return output


def classify_original(rows):
    complete=len(rows)==2 and all(row['complete'] and row['identity_ok'] for row in rows)
    missing=any(row['missing_series'] for row in rows)
    invalid=any(row['invalid_series'] for row in rows) or (not complete and not missing)
    u=max(row['rms_g'] for row in rows) if complete else np.nan
    return dict(complete=complete,u=u,margin=.10-u if complete else np.nan,
                pass_flag=bool(u<=.10) if complete else None,invalid=bool(invalid),missing=bool(missing),
                reason='' if complete else 'required_reference_scale_or_series_invalid_or_missing')


def saved_boolean(value):
    if value is None or str(value) in ('NA','nan',''):
        return None
    if str(value) in ('True','true','1'):
        return True
    if str(value) in ('False','false','0'):
        return False
    raise ValueError('invalid stored boolean')


def summary_rows(endpoints):
    output=[]
    fields=('same_rate','moved_rate','q_changed_rate','q_changed_same_rate','same_given_q_changed_rate',
            'mean_g','rms_g','mean_abs_g','abs_g_p95','abs_g_max','mean_h','rms_h','mean_d','rms_d','cross_term')
    for dist in DISTRIBUTIONS:
        for endpoint in ENDPOINTS:
            rows=[row for row in endpoints if row['distribution']==dist and row['endpoint']==endpoint]
            complete=[row for row in rows if row['complete']]
            result=dict(distribution=dist,endpoint=endpoint,planned_N=30,observed_N=len(rows),complete_N=len(complete),
                        invalid_N=sum(row['invalid_series']>0 for row in rows),
                        missing_N=30-len(rows)+sum(row['missing_series']>0 for row in rows))
            for prefix,selected in (('',complete if len(complete)==30 else []),('conditional_',complete)):
                for field in fields:
                    values=np.asarray([row[field] for row in selected],dtype=float)
                    finite=values[np.isfinite(values)]
                    result[prefix+'mean_'+field]=float(finite.mean()) if finite.size else np.nan
                    result[prefix+field+'_defined_N']=len(finite)
                values=np.asarray([row['rms_g'] for row in selected])
                for suffix,q in (('p50',.5),('p95',.95)):
                    result[prefix+'rms_g_'+suffix]=float(np.quantile(values,q,method='linear')) if values.size else np.nan
                result[prefix+'rms_g_max']=float(values.max()) if values.size else np.nan
                moved=[row['moved_abs_g_mean'] for row in selected if np.isfinite(row['moved_abs_g_mean'])]
                result[prefix+'mean_moved_abs_g_mean']=float(np.mean(moved)) if moved else np.nan
                result[prefix+'moved_distance_defined_N']=len(moved)
            for name in ('same_count','down_count','up_count','q_changed_same_count'):
                result[name]=sum(row[name] for row in rows)
            output.append(result)
    return output


def rank_representatives(originals):
    examples=[]
    for dist in DISTRIBUTIONS:
        rows=[row for row in originals if row['distribution']==dist and row['complete']]
        for rank,row in enumerate(sorted(rows,key=lambda item:(item['u'],item['master_seed'],item['outer_id'])),1):
            row['rank_u']=rank
        if not rows:
            continue
        minimum=min(rows,key=lambda item:(item['u'],item['master_seed'],item['outer_id']))
        maximum=min(rows,key=lambda item:(-item['u'],item['master_seed'],item['outer_id']))
        for label,row in (('minimum',minimum),('maximum',maximum)):
            examples.append(dict(selection=label,**{field:row[field] for field in
                            ('distribution','master_seed','outer_id','sample_sha256','u','margin','rank_u')}))
    return examples


def classification_rows(originals):
    identities=[]
    for row in originals:
        old=row['m11_pass'] if row['m11_complete'] else None
        new=row['pass_flag'] if row['complete'] else None
        category=f'{int(old)}{int(new)}' if old is not None and new is not None else 'unknown'
        identities.append({field:row[field] for field in ('distribution','master_seed','outer_id','sample_sha256')} |
                          dict(m11_pass=old,m12_pass=new,category=category,reason=row['reason']))
    cells=[]
    for dist in DISTRIBUTIONS:
        for old in (False,True):
            for new in (False,True):
                selected=[row for row in identities if row['distribution']==dist and row['m11_pass'] is old and row['m12_pass'] is new]
                cells.append(dict(distribution=dist,m11_pass=old,m12_pass=new,count=len(selected),
                                  ids=';'.join(f"{row['master_seed']}:{row['outer_id']}" for row in selected)))
    return cells,identities


def verify_table(expected,path,key_fields):
    actual=read_csv(path)
    if len(expected)!=len(actual):
        raise ValueError(f'{path.name}: row count differs')
    fields=set().union(*(row.keys() for row in expected))
    if any(set(row)!=fields for row in actual):
        raise ValueError(f'{path.name}: CSV fields differ')
    for index,(first,second) in enumerate(zip(expected,actual)):
        for field in key_fields:
            if str(first[field])!=second[field]:
                raise ValueError(f'{path.name}: row identity/order differs')
        for field in fields:
            value=first.get(field)
            # The frozen CSV writer encodes every nonfinite number as NA.
            if isinstance(value,(float,np.floating)) and not np.isfinite(value):
                value=None
            assert_value(value,second[field],f'{path.name}/{index}/{field}')


def audit(root: Path,output: Path):
    root,output=Path(root),Path(output)
    manifest,receipt,source_hashes=verify_receipt(root,output)
    m11=root/'evidence/rq_m11'
    old_rows=[row for row in read_csv(m11/'original_metrics.csv') if row['method']=='bca']
    index={(row['distribution'],int(row['master_seed']),int(row['outer_id'])):row for row in old_rows}
    expected_ids={(dist,seed,outer) for dist in DISTRIBUTIONS for seed in range(3) for outer in range(10)}
    if set(index)!=expected_ids or len(old_rows)!=60:
        raise ValueError('accepted M11 BCa original IDs differ')
    records=[]
    original_hashes=set()
    panel_cells={}
    # All sixty input identities and source bridges are checked before mapping.
    for dist in DISTRIBUTIONS:
        for seed in range(3):
            panel=load_npz(root/'evidence/rq_m10/panels'/f'{dist}_seed{seed}.npz')
            original=load_npz(root/'evidence/rq_m10/originals'/f'{dist}_seed{seed}.npz')
            if panel['samples'].shape!=(10,5) or panel['samples'].dtype!=np.float64 or panel['samples'].tobytes()!=original['samples'].tobytes():
                raise ValueError('M10 original/panel storage differs')
            if not np.array_equal(panel['outer_id'],np.arange(10)) or panel['outer_id'].dtype!=np.uint16:
                raise ValueError('M10 outer IDs differ')
            if tuple(panel['methods'].astype(str))!=('t','percentile','bca'):
                raise ValueError('M10 method axis differs')
            if not np.array_equal(panel['budget_repeat_id'],np.arange(20)) or not np.array_equal(panel['stream_id'],np.arange(120)):
                raise ValueError('M10 r/k axis differs')
            if panel['budget_repeat_id'].dtype!=np.uint8 or panel['stream_id'].dtype!=np.uint8:
                raise ValueError('M10 r/k storage differs')
            for field,value in dict(distribution=dist,master_seed=seed,n=5,B=4999,alpha=.05,R=20,streams_per_budget=120).items():
                if scalar(panel,field)!=value:
                    raise ValueError(f'M10 fixed metadata differs: {field}')
            panel_cells[(dist,seed)]=panel
            for outer in range(10):
                identity=(dist,seed,outer)
                reference,comparison,ordered=validate_source_original(root,identity,panel)
                digest_value=sample_hash(panel['samples'][outer])
                if digest_value in original_hashes or index[identity]['sample_sha256']!=digest_value:
                    raise ValueError('M11 original row hash/duplicate differs')
                original_hashes.add(digest_value)
                if original['sample_sha256'][outer].decode()!=digest_value:
                    raise ValueError('M10 original hash differs')
                records.append((identity,reference,comparison,index[identity]))
    stems={f'{dist}_seed{seed}_outer{outer}.npz' for dist,seed,outer in expected_ids}
    if {path.name for path in (output/'series').glob('*.npz')}!=stems:
        raise ValueError('fixed sixty output series filenames differ')
    output_hashes={}
    endpoints,originals,reasons,completed=[],[],[],[]
    for (dist,seed,outer),reference,comparison,old in records:
        identity=dict(distribution=dist,master_seed=seed,outer_id=outer,sample_sha256=old['sample_sha256'])
        data=map_original(reference,comparison)
        metadata=dict(**identity,question_id='RQ-M12',source_question_id='RQ-M11',n=5,B=4999,R=20,K=120,
                      endpoints=np.asarray(ENDPOINTS),budget_repeat_id=np.repeat(np.arange(20,dtype=np.uint8),120),
                      stream_id=np.tile(np.arange(120,dtype=np.uint8),20))
        filename=f'{dist}_seed{seed}_outer{outer}.npz'
        path=output/'series'/filename
        assert_arrays({**data,**metadata},load_npz(path),filename,
                      exact_fields=('A','C','Y','q_B','q_star','percentile_A','percentile_C','percentile_q_B','percentile_q_star',
                                    'support','frequency','cumulative','selected_index','A_index','movement_code','q_changed','q_changed_same'))
        output_hashes['series/'+filename]=digest(path)
        local=[]
        for ei,endpoint in enumerate(ENDPOINTS):
            metric=endpoint_metrics(data,ei)
            endpoints.append(dict(**identity,endpoint=endpoint,**metric))
            local.append(metric)
            for code,label in enumerate(data['reason_labels']):
                count=int(np.count_nonzero(data['reason_code'][:,ei]==code))
                if count:
                    reasons.append(dict(**identity,endpoint=endpoint,status='valid' if code==0 else 'missing' if label=='not_observed' else 'invalid',
                                        reason=str(label),series=count))
        old_e=float(old['e']) if old['e'] not in ('','NA') else np.nan
        originals.append(dict(**identity,**classify_original(local),m11_complete=saved_boolean(old['complete']) is True,
                              m11_pass=saved_boolean(old['pass_flag']),m11_e=old_e,lower_r=local[0]['rms_g'],upper_r=local[1]['rms_g'],
                              rank_u=None,source_warning_series=int(np.count_nonzero(data['observed']&data['source_warning']))))
        completed.append(identity)
    representatives=rank_representatives(originals)
    cells,ids=classification_rows(originals)
    groups={dist:population_decision([row for row in originals if row['distribution']==dist]) for dist in DISTRIBUTIONS}
    status=joint_status([groups[dist]['status'] for dist in DISTRIBUTIONS])
    decision=dict(question_id='RQ-M12',hypothesis='H12-Q',computed_status=status,status=status,
                  groups=groups,limit_standard_errors=.10,required_fraction=.90,planned_N_per_group=30,required_pass_count=27,
                  series_per_original=2400,reference='fixed_M11_binary64_conditional_inverseCDF',verification_required=True,
                  classification=cells,classification_unknown_ids=[{key:row[key] for key in ('distribution','master_seed','outer_id')} for row in ids if row['category']=='unknown'],
                  interpretation='Fixed saved-data descriptive classification; public result requires completed run and passed independent audit. No causal attribution/population guarantee.')
    tables=(('original_metrics.csv',originals,('distribution','master_seed','outer_id')),
            ('endpoint_metrics.csv',endpoints,('distribution','master_seed','outer_id','endpoint')),
            ('summary.csv',summary_rows(endpoints),('distribution','endpoint')),
            ('classification.csv',cells,('distribution',)),('classification_ids.csv',ids,('distribution','master_seed','outer_id')),
            ('reasons.csv',reasons,('distribution','master_seed','outer_id','endpoint')))
    for filename,rows,keys in tables:
        verify_table(rows,output/filename,keys)
        output_hashes[filename]=digest(output/filename)
    assert_value(json_safe(decision),read_json(output/'decision.json'),'decision')
    assert_value(json_safe(dict(rule='u min/max pergroup; ties seed,outer ascending',examples=representatives)),
                 read_json(output/'representatives.json'),'representatives')
    resource=read_json(output/'analysis_resources.json')
    expected_resource=dict(question_id='RQ-M12',status='completed',completed_originals=60,completed_ids=completed,
                           saved_common_series=144000,BCa_endpoints=288000,new_originals=0,new_bootstrap_series=0,
                           PRNG_replays=0,M11_analysis_reruns=0,
                           scope='Internal mapping/comparison/persistence loop only; full input validation/CLI/RSS/storage are measured by outer launcher.')
    if set(resource)!=set(expected_resource)|{'analysis_loop_seconds'}:
        raise ValueError('analysis resource schema differs')
    for field,expected in expected_resource.items():
        assert_value(expected,resource[field],'analysis_resources/'+field)
    if not isinstance(resource['analysis_loop_seconds'],(int,float)) or not np.isfinite(resource['analysis_loop_seconds']) or resource['analysis_loop_seconds']<0:
        raise ValueError('analysis loop measurement invalid')
    for name in ('decision.json','representatives.json','analysis_resources.json','input_receipt.json','manifest.json'):
        output_hashes[name]=digest(output/name)
    archived=0
    for (dist,seed),panel in panel_cells.items():
        for stream in (0,29,30,59,60,119):
            verify_archived_stream(root/'evidence/rq_m10/streams'/f'{dist}_seed{seed}_outer0_repeat0_stream{stream}.npz',panel,dist,seed,stream)
            archived+=1
    return json_safe(dict(status='passed',mismatch_count=0,question_id='RQ-M12',originals=60,saved_common_series=144000,
                          BCa_endpoints=288000,reconstructed_finite_streams=archived,groups=groups,computed_status=status,
                          table_counts={name:len(rows) for name,rows,_ in tables},preanalysis_commit=receipt['preanalysis_commit'],
                          source_sha256=source_hashes,output_sha256=output_hashes,
                          limitations=['All288000 BCa endpoints audited from saved inputs; no RNG generation or replay.',
                                       'Existing36 archived indices/means were reconstructed using the accepted independent M11 finite-B primitive; remaining143964 generation processes were not reconstructed.',
                                       'M12 selection uses exact integer ranks on sorted stored3125 means, not the primary inverse-CDF/aggregation/decision functions. No whole M11 analysis or enumeration rerun.',
                                       'Shared primitives are NumPy binary64 arithmetic/linear descriptive quantiles and the disclosed independent finite-B checker. No causal contribution, population guarantee, or external peer review.']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    root=args.root.resolve()
    output=args.output.resolve() if args.output else root/'evidence/rq_m12'
    destination=output/'independent_audit.json'
    if destination.exists():
        raise FileExistsError('independent audit output already exists; preserve the recorded attempt')
    try:
        report=audit(root,output)
        code=0
    except Exception as error:
        report=dict(status='failed',mismatch_count=1,question_id='RQ-M12',error_type=type(error).__name__,
                    error=str(error) if isinstance(error,(ValueError,ArithmeticError)) else 'audit could not verify saved inputs/outputs')
        code=1
    with destination.open('x',encoding='utf-8') as handle:
        json.dump(json_safe(report),handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=report['status'],mismatch_count=report['mismatch_count'])))
    return code


if __name__=='__main__':
    raise SystemExit(main())
