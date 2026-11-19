"""Primary RQ-M12 worker; the outer launcher verifies immutable receipts first.

Only synthetic fixtures may call this before the preanalysis commit. No random
sampling, M11 enumeration or input modification is performed by this module.
"""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from mathscape import probability_movement as engine

DISTRIBUTIONS = ('gamma', 'lognormal')
METHODS = ('bca', 'percentile')
ENDPOINTS = ('lower', 'upper')
ACCEPTED_HEAD = '0deb14fa313dc27380afbb43e6b132386aa40206'
FLOAT_FIELDS = ('saved_endpoint', 'saved_q', 'delta', 'saved_z0', 'saved_a', 'saved_tau_low', 'saved_tau_high')


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {field: data[field] for field in data.files}


def _scalar(data, field):
    value = np.asarray(data[field])
    if value.shape != ():
        raise ValueError(f'metadata must be scalar: {field}')
    return value.item()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    if value is None or str(value) in ('NA', 'nan', ''):
        return None
    raise ValueError('invalid saved boolean')


def _number(value):
    if value is None or str(value) in ('NA', ''):
        return np.nan
    return float(value)


def _hash(sample):
    return sha256(np.asarray(sample, dtype='<f8').tobytes(order='C')).hexdigest()


def validate_inputs(root):
    """Read and validate all60 identities/axes/bridges before the first q map.

    Git/hash allowlists and full raw provenance are verified by the outer
    launcher. Numerical-invalid and missing slots are preserved, not rejected
    as identity/schema corruption and not replaced with zero.
    """
    root = Path(root)
    m11 = root / 'evidence/rq_m11'
    m10 = root / 'evidence/rq_m10'
    old_rows = [row for row in _csv(m11 / 'original_metrics.csv') if row['method'] == 'bca']
    old_index = {(row['distribution'], int(row['master_seed']), int(row['outer_id'])): row for row in old_rows}
    _require(len(old_rows) == len(old_index) == 60, 'M11 original metric IDs missing/duplicated')
    expected = {(d, s, o) for d in DISTRIBUTIONS for s in (0,1,2) for o in range(10)}
    _require(set(old_index) == expected, 'M11 original metric IDs differ')
    records = []
    hashes = set()
    for distribution in DISTRIBUTIONS:
        for seed in (0,1,2):
            name = f'{distribution}_seed{seed}.npz'
            original = _load(m10 / 'originals' / name)
            panel = _load(m10 / 'panels' / name)
            _require(original['samples'].shape == (10,5) and original['samples'].dtype == np.float64,
                     'M10 original sample shape/dtype differs')
            _require(panel['samples'].shape == (10,5) and panel['samples'].dtype == np.float64 and
                     panel['samples'].tobytes() == original['samples'].tobytes(), 'M10 original/panel bytes differ')
            _require(np.array_equal(panel['outer_id'], np.arange(10)) and panel['outer_id'].dtype == np.uint16,
                     'M10 original axis differs')
            _require(list(panel['methods'].astype(str)) == ['t','percentile','bca'], 'M10 method order differs')
            _require(np.array_equal(panel['budget_repeat_id'], np.arange(20)) and panel['budget_repeat_id'].dtype == np.uint8 and
                     np.array_equal(panel['stream_id'], np.arange(120)) and panel['stream_id'].dtype == np.uint8,
                     'M10 r/k axes differ')
            for field, value in dict(distribution=distribution, master_seed=seed, n=5, B=4999, alpha=.05,
                                     R=20, streams_per_budget=120).items():
                _require(_scalar(panel, field) == value, 'M10 fixed metadata differs: ' + field)
            for outer in range(10):
                stem = f'{distribution}_seed{seed}_outer{outer}.npz'
                ref = _load(m11 / 'references' / stem)
                comp = _load(m11 / 'comparisons' / stem)
                sample = original['samples'][outer]
                digest = _hash(sample)
                _require(digest not in hashes, 'duplicate source original')
                hashes.add(digest)
                old = old_index[(distribution, seed, outer)]
                _require(old['sample_sha256'] == digest and original['sample_sha256'][outer].decode() == digest and
                         panel['sample_sha256'][outer].decode() == digest, 'M10/M11 original hash bridge differs')
                for data in (ref, comp):
                    for field, value in dict(distribution=distribution, master_seed=seed, outer_id=outer,
                        sample_sha256=digest, n=5, B=4999, R=20, K=120,
                        source_question_id='RQ-M10', question_id='RQ-M11').items():
                        _require(_scalar(data, field) == value, 'M11 identity metadata differs: ' + field)
                    _require(tuple(data['methods'].astype(str)) == METHODS and tuple(data['endpoints'].astype(str)) == ENDPOINTS,
                             'M11 method/endpoint axes differ')
                engine.validate_distribution(ref['support'], ref['frequency'], ref['cumulative'])
                _require(ref['means'].shape == (3125,) and ref['means'].dtype == np.float64 and np.isfinite(ref['means']).all(),
                         'saved3125 means shape/dtype differs')
                support, counts = np.unique(ref['means'], return_counts=True)
                _require(np.array_equal(support, ref['support']) and np.array_equal(counts, ref['frequency']),
                         'saved means/support/frequency bridge differs')
                for field in ('q','E_cdf','atom_index'):
                    _require(ref[field].shape == (2,2), 'M11 reference axis differs: ' + field)
                _require(ref['q'].dtype == ref['E_cdf'].dtype == np.float64 and ref['atom_index'].dtype == np.int64,
                         'M11 reference numeric dtype differs')
                _require(ref['valid'].shape == (2,) and ref['valid'].dtype == bool and ref['reason'].shape == (2,),
                         'M11 reference masks/reasons differ')
                for field in FLOAT_FIELDS:
                    shape = (2400,2,2) if field in ('saved_endpoint','saved_q','delta') else (2400,2)
                    _require(comp[field].shape == shape and comp[field].dtype == np.float64,
                             'M11 saved numerical axis/dtype differs: ' + field)
                for field in ('saved_valid','saved_warning'):
                    _require(comp[field].shape == (2400,2) and comp[field].dtype == bool, 'M11 method mask differs')
                _require(comp['observed'].shape == (2400,) and comp['observed'].dtype == bool and
                         comp['error_valid'].shape == (2400,2,2) and comp['error_valid'].dtype == bool,
                         'M11 observed/error masks differ')
                _require(comp['saved_reason'].shape == (2400,2) and comp['saved_reason'].dtype.kind == 'U',
                         'M11 source reasons differ')
                _require(np.array_equal(comp['budget_repeat_id'], np.repeat(np.arange(20),120)) and
                         np.array_equal(comp['stream_id'], np.tile(np.arange(120),20)) and
                         comp['budget_repeat_id'].dtype == comp['stream_id'].dtype == np.uint8,
                         'M11 saved rk order differs')
                _require(np.array_equal(comp['observed'], panel['observed_streams'][outer].reshape(-1)),
                         'M10/M11 observed mask differs')
                reasons = panel['reason_labels'][panel['reason_codes'][outer]]
                for mi, source_mi in enumerate((2,1)):
                    for endpoint, field in enumerate(('lower','upper')):
                        _require(np.array_equal(comp['saved_endpoint'][:,mi,endpoint], panel[field][outer,:,source_mi,:].reshape(-1), equal_nan=True),
                                 'M10/M11 saved endpoint bridge differs')
                    for endpoint, field in enumerate(('q_low','q_high')):
                        _require(np.array_equal(comp['saved_q'][:,mi,endpoint], panel[field][outer,:,source_mi,:].reshape(-1), equal_nan=True),
                                 'M10/M11 saved probability bridge differs')
                    for target, field in (('saved_valid','valid'),('saved_warning','tail_warning'),
                        ('saved_z0','z0'),('saved_a','a'),('saved_tau_low','tau_low'),('saved_tau_high','tau_high')):
                        _require(np.array_equal(comp[target][:,mi], panel[field][outer,:,source_mi,:].reshape(-1), equal_nan=True),
                                 'M10/M11 saved diagnostic bridge differs: ' + target)
                    _require(np.array_equal(comp['saved_reason'][:,mi], reasons[:,source_mi,:].reshape(-1)),
                             'M10/M11 saved reason bridge differs')
                with np.errstate(over='ignore', invalid='ignore'):
                    scale = np.std(sample, ddof=1)/np.sqrt(5)
                if np.all(sample == sample[0]):
                    scale = np.nan
                _require(np.isclose(_scalar(ref,'se'), scale, rtol=engine.NUMERIC_RTOL,
                                    atol=engine.NUMERIC_ATOL, equal_nan=True), 'M11 sample SE differs')
                records.append(dict(distribution=distribution, master_seed=seed, outer_id=outer,
                    sample_sha256=digest, reference=ref, comparison=comp,
                    m11_complete=_bool(old['complete']) is True, m11_pass=_bool(old['pass_flag']),
                    m11_e=_number(old['e'])))
    _require(len(records) == 60, 'all60 inputs required before probability mapping')
    return records


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list,tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: 'NA' if value is None or isinstance(value,(float,np.floating)) and not np.isfinite(value) else value
                             for key,value in row.items()})


def save_npz(path, data):
    arrays = {key: np.asarray(value) for key,value in data.items()}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError('object arrays forbidden')
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def endpoint_summary(rows):
    output = []
    metrics = ('same_rate','moved_rate','q_changed_rate','q_changed_same_rate','same_given_q_changed_rate',
               'mean_g','rms_g','mean_abs_g','abs_g_p95','abs_g_max','mean_h','rms_h','mean_d','rms_d','cross_term')
    for distribution in DISTRIBUTIONS:
        for endpoint in ENDPOINTS:
            group = [row for row in rows if row['distribution']==distribution and row['endpoint']==endpoint]
            complete = [row for row in group if row['complete']]
            row = dict(distribution=distribution, endpoint=endpoint, planned_N=30,
                observed_N=len(group), complete_N=len(complete), invalid_N=sum(r['invalid_series']>0 for r in group),
                missing_N=30-len(group)+sum(r['missing_series']>0 for r in group))
            for prefix, selected in (('',complete if len(complete)==30 else []),('conditional_',complete)):
                for field in metrics:
                    values = np.asarray([r[field] for r in selected], dtype=float)
                    finite = values[np.isfinite(values)]
                    # Conditional probabilities can be undefined for q-change0.
                    row[prefix+'mean_'+field] = float(finite.mean()) if len(finite) else np.nan
                    row[prefix+field+'_defined_N'] = len(finite)
                rms = np.asarray([r['rms_g'] for r in selected], dtype=float)
                row[prefix+'rms_g_p50'] = float(np.quantile(rms,.5,method='linear')) if len(rms) else np.nan
                row[prefix+'rms_g_p95'] = float(np.quantile(rms,.95,method='linear')) if len(rms) else np.nan
                row[prefix+'rms_g_max'] = float(rms.max()) if len(rms) else np.nan
                moved = [r['moved_abs_g_mean'] for r in selected if np.isfinite(r['moved_abs_g_mean'])]
                row[prefix+'mean_moved_abs_g_mean'] = float(np.mean(moved)) if moved else np.nan
                row[prefix+'moved_distance_defined_N'] = len(moved)
            row.update(same_count=sum(r['same_count'] for r in group), down_count=sum(r['down_count'] for r in group),
                       up_count=sum(r['up_count'] for r in group), q_changed_same_count=sum(r['q_changed_same_count'] for r in group))
            output.append(row)
    return output


def classifications(originals):
    ids = []
    cells = []
    for row in originals:
        old = row['m11_pass'] if row['m11_complete'] else None
        new = row['pass_flag'] if row['complete'] else None
        category = f'{int(old)}{int(new)}' if old is not None and new is not None else 'unknown'
        ids.append({key:row[key] for key in ('distribution','master_seed','outer_id','sample_sha256')} |
                   dict(m11_pass=old,m12_pass=new,category=category,reason=row['reason']))
    for distribution in DISTRIBUTIONS:
        for old in (False,True):
            for new in (False,True):
                selected = [row for row in ids if row['distribution']==distribution and row['m11_pass'] is old and row['m12_pass'] is new]
                cells.append(dict(distribution=distribution,m11_pass=old,m12_pass=new,count=len(selected),
                                  ids=';'.join(f"{r['master_seed']}:{r['outer_id']}" for r in selected)))
    return cells, ids


def representatives(originals):
    result = []
    for distribution in DISTRIBUTIONS:
        group = [row for row in originals if row['distribution']==distribution and row['complete']]
        for rank,row in enumerate(sorted(group,key=lambda r:(r['u'],r['master_seed'],r['outer_id'])),1):
            row['rank_u'] = rank
        if not group:
            continue
        for selection,key in (('minimum',lambda r:(r['u'],r['master_seed'],r['outer_id'])),
                              ('maximum',lambda r:(-r['u'],r['master_seed'],r['outer_id']))):
            row = min(group,key=key)
            result.append(dict(selection=selection,**{field:row[field] for field in
                ('distribution','master_seed','outer_id','sample_sha256','u','margin','rank_u')}))
    return result


def analyze_records(records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'series').mkdir(exist_ok=False)
    endpoints, originals, reasons = [], [], []
    completed = []
    started = perf_counter()
    persisted = False
    try:
        for record in records:
            identity = {key:record[key] for key in ('distribution','master_seed','outer_id','sample_sha256')}
            arrays = engine.analyze_sample(record['reference'],record['comparison'])
            metadata = dict(**identity,question_id='RQ-M12',source_question_id='RQ-M11',n=5,B=4999,R=20,K=120,
                endpoints=np.asarray(ENDPOINTS),budget_repeat_id=np.repeat(np.arange(20,dtype=np.uint8),120),
                stream_id=np.tile(np.arange(120,dtype=np.uint8),20))
            filename = f"{record['distribution']}_seed{record['master_seed']}_outer{record['outer_id']}.npz"
            save_npz(output/'series'/filename,{**arrays,**metadata})
            local = []
            for ei, endpoint in enumerate(ENDPOINTS):
                metric = engine.endpoint_metrics(arrays,ei)
                endpoints.append(dict(**identity,endpoint=endpoint,**metric))
                local.append(metric)
                codes = arrays['reason_code'][:,ei]
                for code,label in enumerate(arrays['reason_labels']):
                    count = int((codes==code).sum())
                    if count:
                        reasons.append(dict(**identity,endpoint=endpoint,
                            status='valid' if code==0 else 'missing' if label=='not_observed' else 'invalid',
                            reason=str(label),series=count))
            originals.append(dict(**identity,**engine.classify_original(local),
                m11_complete=record['m11_complete'],m11_pass=record['m11_pass'],m11_e=record['m11_e'],
                lower_r=local[0]['rms_g'],upper_r=local[1]['rms_g'],rank_u=None,
                source_warning_series=int((arrays['observed']&arrays['source_warning']).sum())))
            completed.append(identity)
        selected = representatives(originals)
        cells, ids = classifications(originals)
        groups = {d:engine.group_decision([r for r in originals if r['distribution']==d]) for d in DISTRIBUTIONS}
        status = engine.joint_status([groups[d]['status'] for d in DISTRIBUTIONS])
        decision = dict(question_id='RQ-M12',hypothesis='H12-Q',computed_status=status,status=status,
            groups=groups,limit_standard_errors=.10,required_fraction=.90,planned_N_per_group=30,
            required_pass_count=27,series_per_original=2400,reference='fixed_M11_binary64_conditional_inverseCDF',
            verification_required=True,classification=cells,
            classification_unknown_ids=[{key:r[key] for key in ('distribution','master_seed','outer_id')} for r in ids if r['category']=='unknown'],
            interpretation='Fixed saved-data descriptive classification; public result requires completed run and passed independent audit. No causal attribution/population guarantee.')
        write_csv(output/'original_metrics.csv',originals)
        write_csv(output/'endpoint_metrics.csv',endpoints)
        write_csv(output/'summary.csv',endpoint_summary(endpoints))
        write_csv(output/'classification.csv',cells)
        write_csv(output/'classification_ids.csv',ids)
        write_csv(output/'reasons.csv',reasons)
        write_json(output/'decision.json',decision)
        write_json(output/'representatives.json',dict(rule='u min/max pergroup; ties seed,outer ascending',examples=selected))
        persisted = True
        return decision
    finally:
        write_json(output/'analysis_resources.json',dict(question_id='RQ-M12',status='completed' if persisted else 'stopped',
            completed_originals=len(completed),completed_ids=completed,analysis_loop_seconds=perf_counter()-started,
            saved_common_series=2400*len(completed),BCa_endpoints=4800*len(completed),
            new_originals=0,new_bootstrap_series=0,PRNG_replays=0,M11_analysis_reruns=0,
            scope='Internal mapping/comparison/persistence loop only; full input validation/CLI/RSS/storage are measured by outer launcher.'))


def analyze(root: Path, output: Path):
    output = Path(output)
    for name in ('manifest.json','input_receipt.json'):
        if not (output/name).is_file():
            raise ValueError('immutable launcher receipt required before analysis')
    return analyze_records(validate_inputs(root),output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root/'evidence/rq_m12'
    result = analyze(root,output)
    print(json.dumps(dict(status='completed',computed_status=result['computed_status'])))


if __name__ == '__main__':
    main()
