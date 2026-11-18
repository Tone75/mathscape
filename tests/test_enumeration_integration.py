"""Actual M11 writer/auditor round trip using only invented deterministic data.

No accepted M10 file is read, no random generator is available, and every
saved resampling index is an explicitly prescribed fixture integer.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from mathscape import matched_moments, small_sample, stability
from scripts import analyze_enumerated_reference as primary
from scripts import audit_enumerated_reference as independent


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


@pytest.fixture(autouse=True)
def no_random_or_real_original_generation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('integration fixture forbids RNG and main originals')
    for name in ('SeedSequence', 'PCG64', 'Generator', 'default_rng'):
        monkeypatch.setattr(np.random, name, forbidden)
    monkeypatch.setattr(small_sample, 'draw_original', forbidden)
    monkeypatch.setattr(small_sample, 'generate_cohort', forbidden)


def _synthetic_source(root):
    source = root / 'invented_m10'
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
        panel = {field: np.full(shape, np.nan) for field in primary.PANEL_FLOAT_FIELDS}
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
                for field in (*primary.PANEL_FLOAT_FIELDS, 'valid', 'tail_warning', 'less_count', 'equal_count'):
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
            for field in primary.PANEL_FLOAT_FIELDS:
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
            reference_accepted_head=independent.ACCEPTED_INPUT)
        _save(source / 'panels' / f'{distribution}_seed{seed}.npz', panel)
    return source, receipt


@pytest.fixture(scope='module')
def synthetic_roundtrip(tmp_path_factory):
    # Module fixture explicitly prohibits RNG even during setup, before the
    # function-scoped prohibition above takes effect.
    with pytest.MonkeyPatch.context() as blocked:
        def forbidden(*args, **kwargs):
            raise AssertionError('synthetic source has no RNG/main generation')
        for name in ('SeedSequence', 'PCG64', 'Generator', 'default_rng'):
            blocked.setattr(np.random, name, forbidden)
        blocked.setattr(small_sample, 'draw_original', forbidden)
        blocked.setattr(small_sample, 'generate_cohort', forbidden)
        root = tmp_path_factory.mktemp('invented_enumeration_only')
        source, original_receipt = _synthetic_source(root)
        _json(source / 'manifest.json', {'source_files_sha256': {}})
        for name in ('hypotheses.json', 'resources.json', 'independent_audit.json'):
            _json(source / name, {'fixture_only': True})
        (source / 'REPORT.md').write_text('Invented deterministic fixture only.\n')
        output = root / 'invented_m11'
        output.mkdir()
        known = {}
        package_init = root / 'mathscape/__init__.py'
        package_init.parent.mkdir()
        package_init.write_text('"""Invented package input, frozen separately from the M10 manifest."""\n')
        package_bytes = package_init.read_bytes()
        from hashlib import sha1
        known['mathscape/__init__.py'] = dict(
            sha256=sha256(package_bytes).hexdigest(),
            git_blob=sha1(b'blob '+str(len(package_bytes)).encode()+b'\0'+package_bytes).hexdigest())
        for path in sorted(source.rglob('*')):
            if path.is_file():
                content = path.read_bytes()
                from hashlib import sha1
                known[str(path.relative_to(root))] = dict(sha256=sha256(content).hexdigest(),
                    git_blob=sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest())
        manifest = dict(accepted_head=primary.ACCEPTED_HEAD, known_inputs=known,
            original_ids_and_receipt=original_receipt['records'], n=5, alpha=.05,
            finite_B=4999, ordered_n5=3125, series_per_original=2400,
            unique_saved_series=144000, new_draws=0, prng_replays=0)
        _json(output / 'manifest.json', manifest)
        _json(output / 'input_receipt.json', dict(accepted_head=primary.ACCEPTED_HEAD,
            manifest_sha256=sha256((output / 'manifest.json').read_bytes()).hexdigest(),
            inputs=known, originals=original_receipt['records'], preanalysis_commit='a'*40))
        primary.analyze(source, output)
    return root, source, output


def test_actual_analyze_and_independent_audit_full_synthetic_saved_schema(synthetic_roundtrip, monkeypatch):
    root, source, output = synthetic_roundtrip
    monkeypatch.setattr(independent, 'ROOT', root)
    result = independent.audit(source, output)
    assert result['status'] == 'passed' and result['mismatch_count'] == 0
    assert result['originals'] == 60 and result['saved_common_streams'] == 144000
    assert result['reconstructed_finite_streams'] == 36
    assert result['table_counts'] == {
        'reference_metrics.csv': 240, 'endpoint_metrics.csv': 240,
        'original_metrics.csv': 120, 'summary.csv': 4}
    summary = json.loads((output / 'hypotheses.json').read_text())
    assert summary['groups']['gamma']['planned_N'] == 30
    assert summary['groups']['gamma']['status'] == 'inconclusive'
    assert summary['groups']['gamma']['missing_N'] == 1
    assert summary['groups']['gamma']['invalid_N'] >= 1
    invalid = _read(output / 'comparisons/gamma_seed2_outer4.npz')
    assert not invalid['saved_valid'][121, 0] and invalid['saved_valid'][121, 1]
    assert not invalid['error_valid'][121, 0].any() and invalid['error_valid'][121, 1].all()
    missing = _read(output / 'comparisons/gamma_seed1_outer3.npz')
    assert not missing['observed'][121] and not missing['error_valid'][121].any()
    assert len(list((output / 'references').glob('*.npz'))) == 60
    assert len(list((output / 'comparisons').glob('*.npz'))) == 60


def test_saved_cdf_atom_requires_exact_equality_not_numeric_tolerance(synthetic_roundtrip, monkeypatch):
    root, source, output = synthetic_roundtrip
    monkeypatch.setattr(independent, 'ROOT', root)
    path = output / 'references/gamma_seed0_outer0.npz'
    old = path.read_bytes()
    payload = _read(path)
    payload['E_cdf'][0, 0] = np.nextafter(payload['E_cdf'][0, 0], np.inf)
    try:
        _save(path, payload)
        with pytest.raises(ValueError, match='E_cdf.*values differ'):
            independent.audit(source, output)
    finally:
        path.write_bytes(old)


@pytest.mark.parametrize('tamper', ['last_panel_id', 'raw_file_hash', 'raw_transform'])
def test_input_corruption_rejected_before_any_reference_enumeration(synthetic_roundtrip, monkeypatch, tamper):
    _, source, output = synthetic_roundtrip
    def forbidden(*args, **kwargs):
        raise AssertionError('input validation must finish before enumeration')
    monkeypatch.setattr(primary.engine, 'build_reference', forbidden)
    if tamper == 'raw_transform':
        load = primary._load
        def altered_read(path):
            payload = load(path)
            if path == source / 'originals/lognormal_seed2.npz':
                payload['raw'][9, 0] += .125
            return payload
        monkeypatch.setattr(primary, '_load', altered_read)
        with pytest.raises(ValueError, match='raw transformation differs'):
            primary.analyze(source, output / 'unreachable_raw_transform')
        return
    path = source / ('panels/lognormal_seed2.npz' if tamper == 'last_panel_id'
                     else 'originals/lognormal_seed2.npz')
    old = path.read_bytes()
    payload = _read(path)
    if tamper == 'last_panel_id':
        payload['outer_id'][-1] = 8
    else:
        payload['raw'][9, 0] += .125
    try:
        _save(path, payload)
        with pytest.raises(ValueError, match='ID differs|hash differs'):
            primary.analyze(source, output / ('unreachable_'+tamper))
    finally:
        path.write_bytes(old)
