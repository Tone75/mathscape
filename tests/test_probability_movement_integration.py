"""M12 end-to-end artificial saved inputs; never use accepted scientific values."""
from hashlib import sha1, sha256
import json
from pathlib import Path

import numpy as np
import pytest

from mathscape import matched_moments, small_sample, stability, enumerated_reference
from scripts import analyze_enumerated_reference as m11_writer
from scripts import audit_enumerated_reference as m11_audit

def _hash(values):
    return sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()

def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        np.savez_compressed(handle, **payload)

def _read(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}

def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))

def _synthetic_source(root):
    source = root / 'evidence/rq_m10'
    for directory in ('originals', 'panels', 'streams'):
        (source / directory).mkdir(parents=True)
    # These are fixed integers, not a PRNG replay or a stochastic pilot.
    rows = np.arange(4999, dtype=np.int64)[:, None]
    powers = np.array([625, 125, 25, 5, 1], dtype=np.int64)
    indices = ((rows // powers) % 5).astype(np.uint8)
    receipt_records, file_hashes, cells = [], {}, []
    for distribution in ('gamma', 'lognormal'):
        for seed in (0, 1, 2):
            base = (np.array([.11, .31, .71, 1.21, 2.41]) if distribution == 'gamma'
                    else np.array([-.91, -.41, .01, .21, .81]))
            raw = np.stack([base + .003 * (seed * 10 + outer) for outer in range(10)]).astype('<f8')
            samples = np.stack([small_sample.transform_raw(row, distribution) for row in raw])
            constants = matched_moments.population_info(distribution)['constants']
            keys = np.array([(0x4D3130, 1 if distribution == 'gamma' else 2, 5, outer, 0, 0)
                             for outer in range(10)], dtype=np.uint32)
            cell = dict(samples=samples, raw=raw,
                sample_sha256=np.array([_hash(row) for row in samples], dtype='S64'),
                raw_sha256=np.array([_hash(row) for row in raw], dtype='S64'),
                outer_id=np.arange(10, dtype=np.uint16), original_keys=keys,
                distribution=distribution, master_seed=seed, n=5, namespace=0x4D3130,
                raw_kind='gamma_y' if distribution == 'gamma' else 'normal_z',
                constant_names=np.array(list(constants)),
                constant_values=np.array(list(constants.values()), dtype='<f8'))
            name = f'{distribution}_seed{seed}.npz'
            _save(source / 'originals' / name, cell)
            file_hashes[name] = sha256((source / 'originals' / name).read_bytes()).hexdigest()
            for outer in range(10):
                receipt_records.append(dict(distribution=distribution, master_seed=seed,
                    outer_id=outer, n=5, namespace=0x4D3130, entropy=seed,
                    spawn_key=keys[outer].tolist(), constants=constants,
                    sample_sha256=_hash(samples[outer]), raw_sha256=_hash(raw[outer])))
            cells.append((distribution, seed, cell))
    receipt = dict(question_id='RQ-M10', original_samples=60, records=receipt_records,
                   original_files_sha256=file_hashes)
    receipt_path = source / 'originals/receipt.json'
    _json(receipt_path, receipt)
    receipt_digest = sha256(receipt_path.read_bytes()).hexdigest()
    shape = (10, 20, 3, 120)
    for distribution, seed, cell in cells:
        panel = {field: np.full(shape, np.nan) for field in m11_writer.PANEL_FLOAT_FIELDS}
        panel.update(valid=np.zeros(shape, bool), tail_warning=np.zeros(shape, bool),
            less_count=np.full(shape, 65535, dtype=np.uint16),
            equal_count=np.full(shape, 65535, dtype=np.uint16),
            reason_codes=np.zeros(shape, dtype=np.uint8),
            observed_streams=np.ones((10, 20, 120), bool))
        labels = ['', 'bca_bias_correction_nonfinite', 'fixture_unobserved']

        def install(outer, records, repeat=slice(None), stream=slice(None)):
            for method_index, method in enumerate(('t', 'percentile', 'bca')):
                record = records[method]
                target = (outer, repeat, method_index, stream)
                for field in (*m11_writer.PANEL_FLOAT_FIELDS, 'valid', 'tail_warning', 'less_count', 'equal_count'):
                    panel[field][target] = record[field]
                panel['reason_codes'][target] = labels.index(record['reason'])

        for outer, sample in enumerate(cell['samples']):
            means = np.mean(sample[indices], axis=1)
            records = small_sample.infer_single(stability.prepare_sample(sample), means)
            install(outer, records)
            if outer == 0:
                for stream in (0, 29, 30, 59, 60, 119):
                    _save(source / 'streams' / f'{distribution}_seed{seed}_outer0_repeat0_stream{stream}.npz',
                        dict(indices=indices, means=means, distribution=distribution,
                            master_seed=seed, outer_id=0, budget_repeat_id=0, stream_id=stream,
                            n=5, B=4999, R=20, streams_per_budget=120, alpha=.05,
                            namespace=0x4D3130, native_indices_dtype='int64',
                            stored_indices_dtype='uint8', sample_sha256=_hash(sample)))
        # Two scientific-data edge cases outside the 36 saved audit streams.
        if distribution == 'gamma' and seed == 2:
            sample = cell['samples'][4]
            invalid_bca = small_sample.infer_single(stability.prepare_sample(sample),
                                                   np.full(4999, sample.max()))
            assert not invalid_bca['bca']['valid'] and invalid_bca['percentile']['valid']
            install(4, invalid_bca, repeat=1, stream=1)
        if distribution == 'gamma' and seed == 1:
            panel['observed_streams'][3, 1, 1] = False
            for field in m11_writer.PANEL_FLOAT_FIELDS:
                panel[field][3, 1, :, 1] = np.nan
            panel['valid'][3, 1, :, 1] = False
            panel['tail_warning'][3, 1, :, 1] = False
            panel['less_count'][3, 1, :, 1] = 65535
            panel['equal_count'][3, 1, :, 1] = 65535
            panel['reason_codes'][3, 1, :, 1] = 2
        panel.update(samples=cell['samples'], sample_sha256=cell['sample_sha256'],
            outer_id=cell['outer_id'], budget_repeat_id=np.arange(20, dtype=np.uint8),
            stream_id=np.arange(120, dtype=np.uint8), methods=np.array(['t', 'percentile', 'bca']),
            reason_labels=np.array(labels), distribution=distribution, master_seed=seed,
            n=5, namespace=0x4D3130, O=10, R=20, streams_per_budget=120,
            B=4999, alpha=.05, cohort='M10', original_receipt_sha256=receipt_digest,
            reference_accepted_head=m11_audit.ACCEPTED_INPUT)
        _save(source / 'panels' / f'{distribution}_seed{seed}.npz', panel)
    return source, receipt


@pytest.fixture(autouse=True)
def forbid_real_random_or_new_originals(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('M12 fixture cannot run RNG or generate main originals')
    for name in ('default_rng', 'SeedSequence', 'Generator', 'PCG64'):
        monkeypatch.setattr(np.random, name, forbidden)
    monkeypatch.setattr(small_sample, 'draw_original', forbidden)
    monkeypatch.setattr(small_sample, 'generate_cohort', forbidden)
    # Fixture construction uses pure primitives on invented arrays, never the
    # old M11 main entry points or a real scientific stage.
    monkeypatch.setattr(m11_writer, 'analyze', forbidden)
    monkeypatch.setattr(m11_writer, 'analyze_records', forbidden)


def _invented_m11(root, receipt):
    """Canonical M11-shaped arrays from prescribed fake n5 data, not its run."""
    source = root / 'evidence/rq_m10'
    saved = root / 'evidence/rq_m11'
    for folder in ('references', 'comparisons'):
        (saved / folder).mkdir(parents=True)
    records = m11_writer.validate_inputs(source)
    originals, endpoints = [], []
    for record in records:
        reference = enumerated_reference.build_reference(record['sample'])
        comparison = enumerated_reference.stream_comparison(reference, m11_writer.saved_arrays(record))
        identity = {name: record[name] for name in ('distribution', 'master_seed', 'outer_id', 'sample_sha256')}
        metadata = dict(**identity, methods=np.array(['bca', 'percentile']), endpoints=np.array(['lower', 'upper']),
            source_question_id='RQ-M10', question_id='RQ-M11', n=5, B=4999, R=20, K=120)
        stem = f"{identity['distribution']}_seed{identity['master_seed']}_outer{identity['outer_id']}.npz"
        _save(saved / 'references' / stem, {**reference, **metadata})
        _save(saved / 'comparisons' / stem, {**comparison, **metadata})
        for mi, method in enumerate(('bca', 'percentile')):
            local = []
            for ei, endpoint in enumerate(('lower', 'upper')):
                measured = enumerated_reference.endpoint_metrics(comparison, mi, ei)
                local.append(measured)
                endpoints.append(dict(**identity, method=method, endpoint=endpoint, **measured))
            originals.append(dict(**identity, method=method, **enumerated_reference.classify_original(local),
                rank_e=None, saved_warning_series=int(comparison['saved_warning'][:, mi].sum()),
                reference_valid=bool(reference['valid'][mi]), reference_reason=str(reference['reason'][mi])))
    m11_writer.rank_and_represent(originals)
    m11_writer._write_csv(saved / 'original_metrics.csv', originals)
    m11_writer._write_csv(saved / 'endpoint_metrics.csv', endpoints)
    _json(saved / 'input_receipt.json', dict(originals=receipt['records']))
    return saved


@pytest.fixture(scope='module')
def artificial_roundtrip(tmp_path_factory):
    # No autouse fixture runs before a module fixture, so explicitly install
    # the same prohibition throughout all synthetic source construction.
    with pytest.MonkeyPatch.context() as blocked:
        def forbidden(*args, **kwargs):
            raise AssertionError('invented M12 integration has no RNG/main-original calls')
        for name in ('default_rng', 'Generator', 'PCG64', 'SeedSequence'):
            blocked.setattr(np.random, name, forbidden)
        for name in ('generate_cohort', 'draw_original'):
            blocked.setattr(small_sample, name, forbidden)
        blocked.setattr(m11_writer, 'analyze', forbidden)
        blocked.setattr(m11_writer, 'analyze_records', forbidden)
        root = tmp_path_factory.mktemp('artificial_probability_movement')
        _, original_receipt = _synthetic_source(root)
        _invented_m11(root, original_receipt)
        output = root / 'evidence/rq_m12'
        output.mkdir()
        sources = {}
        for question in ('rq_m10', 'rq_m11'):
            for path in sorted((root / 'evidence' / question).rglob('*')):
                if path.is_file():
                    content = path.read_bytes()
                    sources[str(path.relative_to(root))] = dict(sha256=sha256(content).hexdigest(),
                        git_blob=sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest())
        manifest = dict(question_id='RQ-M12',
            accepted_head='0deb14fa313dc27380afbb43e6b132386aa40206',
            source_inputs=sources, originals=original_receipt['records'], sample_size=5,
            B=4999, alpha=.05, originals_per_population=30,
            paired_streams_per_original=2400, total_distinct_saved_series=144000,
            BCa_endpoint_observations=288000)
        _json(output / 'manifest.json', manifest)
        _json(output / 'input_receipt.json', dict(question_id='RQ-M12',
            accepted_head=manifest['accepted_head'], preanalysis_commit='b'*40,
            manifest_sha256=sha256((output/'manifest.json').read_bytes()).hexdigest(),
            inputs=sources, originals=original_receipt['records']))
        from scripts import analyze_probability_movement as main
        main.analyze(root, output)
    return root, output


def test_actual_main_and_independent_audit_artificial_saved_schema(artificial_roundtrip, monkeypatch):
    root, output = artificial_roundtrip
    from mathscape import probability_movement
    from scripts import analyze_probability_movement as main
    from scripts import audit_probability_movement as auditor
    def forbidden(*args, **kwargs):
        raise AssertionError('independent auditor cannot reuse primary numeric/decision functions')
    for name in ('cumulative_index','analyze_sample','endpoint_metrics','classify_original','group_decision','joint_status'):
        monkeypatch.setattr(probability_movement, name, forbidden)
    monkeypatch.setattr(main, 'analyze', forbidden)
    report = auditor.audit(root, output)
    assert report['status'] == 'passed' and report['mismatch_count'] == 0
    assert report['originals'] == 60 and report['saved_common_series'] == 144000
    assert report['BCa_endpoints'] == 288000 and report['reconstructed_finite_streams'] == 36
    assert report['table_counts']['original_metrics.csv'] == 60
    assert report['table_counts']['endpoint_metrics.csv'] == 120
    assert report['table_counts']['summary.csv'] == 4
    assert report['table_counts']['classification.csv'] == 8
    assert report['table_counts']['classification_ids.csv'] == 60
    json.dumps(report, allow_nan=False)
    for relative, expected in report['source_sha256'].items():
        assert sha256((root/relative).read_bytes()).hexdigest() == expected
    for relative, expected in report['output_sha256'].items():
        assert sha256((output/relative).read_bytes()).hexdigest() == expected


def test_support_value_one_ulp_corruption_is_not_hidden_by_arithmetic_tolerance(artificial_roundtrip):
    root, output = artificial_roundtrip
    from scripts import audit_probability_movement as auditor
    path = output/'series/gamma_seed0_outer0.npz'
    content = path.read_bytes()
    data = _read(path)
    data['C'][0,0] = np.nextafter(data['C'][0,0], np.inf)
    try:
        _save(path, data)
        with pytest.raises(ValueError, match='/C: values differ'):
            auditor.audit(root, output)
    finally:
        path.write_bytes(content)


def test_full_artificial_counts_missing_and_method_local_control(artificial_roundtrip):
    _, output = artificial_roundtrip
    assert len(list((output/'series').glob('*.npz'))) == 60
    decision = json.loads((output/'decision.json').read_bytes())
    group = decision['groups']['gamma']
    assert group['planned_N'] == 30 and group['observed_N'] == 30
    assert group['missing_N'] == 1 and group['invalid_N'] == 1
    assert group['unknown_N'] == 2 and group['status'] == 'inconclusive'
    assert len(decision['classification_unknown_ids']) == 2
    data = _read(output/'series/gamma_seed2_outer4.npz')
    assert not data['valid'][121].any() and data['percentile_valid'][121].all()
    assert np.all(data['percentile_g'][data['percentile_valid']] == 0)
    missing = _read(output/'series/gamma_seed1_outer3.npz')
    assert not missing['observed'][121] and not missing['valid'][121].any()
    assert all(missing['reason_labels'][missing['reason_code'][121]] == 'not_observed')
    resources = json.loads((output/'analysis_resources.json').read_bytes())
    assert resources['completed_originals'] == 60
    assert resources['saved_common_series'] == 144000 and resources['BCa_endpoints'] == 288000
    assert all(resources[name] == 0 for name in ('new_originals','new_bootstrap_series','PRNG_replays','M11_analysis_reruns'))


@pytest.mark.parametrize('damage', ['last_identity', 'saved_probability_bridge'])
def test_last_input_corruption_stops_before_any_probability_mapping(artificial_roundtrip, tmp_path, monkeypatch, damage):
    root, _ = artificial_roundtrip
    from scripts import analyze_probability_movement as main
    def forbidden(*args, **kwargs):
        raise AssertionError('all60 input validation must finish before first probability mapping')
    monkeypatch.setattr(main.engine, 'analyze_sample', forbidden)
    path = root/'evidence/rq_m11/comparisons/lognormal_seed2_outer9.npz'
    content = path.read_bytes()
    changed = _read(path)
    if damage == 'last_identity':
        changed['outer_id'] = np.asarray(8)
    else:
        changed['saved_q'][0,0,0] = np.nextafter(changed['saved_q'][0,0,0], 1.0)
    output = tmp_path/'must_not_produce_science'
    output.mkdir()
    _json(output/'manifest.json', {})
    _json(output/'input_receipt.json', {})
    try:
        _save(path, changed)
        with pytest.raises(ValueError, match='identity metadata differs|probability bridge differs'):
            main.analyze(root, output)
        assert not (output/'series').exists()
    finally:
        path.write_bytes(content)
