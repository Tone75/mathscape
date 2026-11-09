"""Fixed fixtures and public references, never a smoke or main experiment."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtri

from mathscape import matched_moments as m05
from mathscape import matched_stability as m06
from mathscape import stability


SAMPLE = np.array([0., .1, .2, .2, .3, .5, .7, .9, 1., 1.2,
                   1.5, 1.8, 2., 2.5, 3., 3.2, 4., 5., 7., 9.])


def test_rng_exact_mapping_and_new_namespaces_no_original_generation():
    assert m06.MAIN_NAMESPACE == 0x4D3036
    assert m06.SMOKE_NAMESPACE == 0x534D3036
    assert m06.DISTRIBUTION_IDS == {"gamma": 1, "lognormal": 2}
    assert not hasattr(m06, "generate_sample")
    for distribution, identifier in m06.DISTRIBUTION_IDS.items():
        actual = m06.rng_for_panel(distribution, 20, 2, 199, 1, 29)
        expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            entropy=2, spawn_key=(0x4D3036, identifier, 20, 199, 1, 29))))
        np.testing.assert_array_equal(actual.bit_generator.random_raw(7), expected.bit_generator.random_raw(7))
    with pytest.raises(ValueError, match="purpose_id"):
        m06.rng_for_panel("gamma", purpose_id=0)
    assert m06.MAIN_NAMESPACE != m05.MAIN_NAMESPACE


@pytest.mark.parametrize("kwargs", [dict(distribution="normal"), dict(distribution=[]),
    dict(n=19), dict(n=True), dict(master_seed=-1), dict(master_seed=3),
    dict(master_seed=np.nan), dict(master_seed=1.), dict(outer_id=-1),
    dict(outer_id=200), dict(inner_id=30), dict(inner_id=True),
    dict(namespace=m05.MAIN_NAMESPACE), dict(namespace=np.inf), dict(purpose_id=2)])
def test_invalid_rng_keys_are_rejected(kwargs):
    arguments = {"distribution": "gamma"} | kwargs
    with pytest.raises(ValueError):
        m06.rng_for_panel(**arguments)


def test_tiny_fixed_sample_indices_int64_generation_uint8_storage_and_order_batch_prefix():
    actual, indices = m06.generate_bootstrap_means(
        SAMPLE, "lognormal", master_seed=1, outer_id=17, inner_id=4,
        Bmax=23, namespace=m06.SMOKE_NAMESPACE, return_indices=True)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        entropy=1, spawn_key=(m06.SMOKE_NAMESPACE, 2, 20, 17, 1, 4))))
    expected_indices = rng.integers(0, 20, size=(23, 20), dtype=np.int64)
    assert indices.dtype == np.uint8
    np.testing.assert_array_equal(indices.astype(np.int64), expected_indices)
    np.testing.assert_array_equal(actual, SAMPLE[expected_indices].mean(axis=1))
    m06.rng_for_panel("gamma", outer_id=3, inner_id=21)
    for limit in (20, 60, 1_000_000):
        repeated = m06.generate_bootstrap_means(
            SAMPLE, "lognormal", master_seed=1, outer_id=17, inner_id=4,
            Bmax=23, namespace=m06.SMOKE_NAMESPACE, max_elements=limit)
        np.testing.assert_array_equal(repeated, actual)
    prefix = m06.generate_bootstrap_means(
        SAMPLE, "lognormal", master_seed=1, outer_id=17, inner_id=4,
        Bmax=9, namespace=m06.SMOKE_NAMESPACE)
    np.testing.assert_array_equal(prefix, actual[:9])


def test_generation_does_not_recalculate_t_or_a_and_explicit_rng_dtype(monkeypatch):
    monkeypatch.setattr(m06, "prepare_sample", lambda *a, **kw: pytest.fail("t/a recomputed"))
    calls = []
    class FixedRng:
        def integers(self, low, high, size, dtype):
            assert (low, high, dtype) == (0, 20, np.int64)
            calls.append(size)
            return np.arange(np.prod(size), dtype=dtype).reshape(size) % 20
    monkeypatch.setattr(m06, "rng_for_panel", lambda *a, **kw: FixedRng())
    means, indices = m06.generate_bootstrap_means(SAMPLE, "gamma", Bmax=7,
                                                  max_elements=60, return_indices=True)
    assert calls == [(3, 20), (3, 20), (1, 20)]
    np.testing.assert_array_equal(means, np.full(7, SAMPLE.mean()))
    np.testing.assert_array_equal(SAMPLE[indices].mean(axis=1), means)


@pytest.mark.parametrize("sample,kwargs", [
    (np.zeros(20), {}), (np.full(20, np.nan), {}), (np.ones(19), {}),
    (np.ones((2, 10)), {}), (np.arange(20, dtype=complex), {}),
    (np.ones(20, dtype=bool), {}), (SAMPLE, {"Bmax": 5000}),
    (SAMPLE, {"Bmax": 0}), (SAMPLE, {"max_elements": 19}),
    (SAMPLE, {"max_elements": 1000001}), (SAMPLE, {"return_indices": 1})])
def test_invalid_samples_or_resource_limits_do_not_draw(sample, kwargs):
    with pytest.raises(ValueError):
        m06.generate_bootstrap_means(sample, "gamma", **kwargs)


def test_shared_4999_reference_matches_scipy_both_prefixes_and_recomputes_z0():
    # One deterministic public-API numerical reference, not population draws.
    reference = stats.bootstrap((SAMPLE,), np.mean, n_resamples=4999,
        method="percentile", confidence_level=.95, rng=np.random.default_rng(74))
    means = reference.bootstrap_distribution
    context = m06.prepare_sample(SAMPLE)
    actual = m06.infer_prefixes(context, means)
    assert m06.prepare_sample is stability.prepare_sample
    assert m06.mean_t_interval is stability.mean_t_interval
    assert m06.welch_interval is stability.welch_interval
    assert set(inspect.signature(m06.infer_prefixes).parameters) == {"context", "means", "B_values"}
    for b in (999, 4999):
        prefix_reference = deepcopy(reference)
        prefix_reference.bootstrap_distribution = means[:b]
        bca = stats.bootstrap((SAMPLE,), np.mean, n_resamples=0, method="BCa",
                             confidence_level=.95, bootstrap_result=prefix_reference)
        for method in m06.METHODS:
            existing = stability.infer_prefixes(context, means, (b,))[b][method]
            for key, value in actual[b][method].items():
                assert (np.isnan(value) and np.isnan(existing[key])
                        if isinstance(value, float) and np.isnan(value) else value == existing[key])
        expected_pct = np.quantile(means[:b], [.025, .975], method="linear")
        assert [actual[b]["percentile"][k] for k in ("lower", "upper")] == pytest.approx(expected_pct)
        assert [actual[b]["bca"][k] for k in ("lower", "upper")] == pytest.approx(
            [bca.confidence_interval.low, bca.confidence_interval.high], abs=1e-12)
        p = (np.count_nonzero(means[:b] < SAMPLE.mean()) + .5*np.count_nonzero(means[:b] == SAMPLE.mean()))/b
        assert actual[b]["bca"]["z0"] == ndtri(p)
        assert actual[b]["bca"]["a"] == context["a"]
        assert actual[b]["t"] == context["t"]
    assert actual[999]["bca"]["z0"] != actual[4999]["bca"]["z0"]


def test_finite_prefix_invalid_suffix_and_tie_convention():
    context = m06.prepare_sample([0., 1., 2.], alpha=.5)
    means = np.array([0., 1., 1., 2., np.nan])
    result = m06.infer_prefixes(context, means, (4, 5))
    assert result[4]["bca"]["valid"] and result[4]["bca"]["z0"] == 0
    assert result[4]["percentile"]["lower"] == .75
    assert not result[5]["bca"]["valid"] and result[5]["t"]["valid"]


def endpoint_fixture():
    lo = np.array([[-1]*4, [-3, -1, .2, 1], [-2, -1, 0, 1]], float)
    hi = np.array([[1]*4, [-2, .2, 1, 2], [-1, 0, 1, 2]], float)
    return {"lower": np.stack([lo, lo]), "upper": np.stack([hi, hi]),
            "valid": np.ones((2, 3, 4), bool), "tail_warning": np.zeros((2, 3, 4), bool)}


def test_metrics_within_original_ddof1_scales_variance_and_sign_partition():
    data = endpoint_fixture()
    data["tail_warning"][:, 2, 0] = True
    actual = m06.original_metrics(data, .5)
    for bi in range(2):
        assert actual["complete"][bi].all()
        np.testing.assert_array_equal(actual["total_inner"][bi], [1, 4, 4])
        np.testing.assert_allclose(actual["C"][bi], [1, .25, .5])
        np.testing.assert_allclose(actual["Mbelow"][bi], [0, .25, .25])
        np.testing.assert_allclose(actual["Mabove"][bi], [0, .5, .25])
        np.testing.assert_allclose(actual["T"][bi], [0, -.25, 0])
        np.testing.assert_allclose(actual["C"][bi]+actual["Mbelow"][bi]+actual["Mabove"][bi], 1)
        assert actual["warning_rate"][bi, 2] == .25
        for mi in (1, 2):
            sd_l, sd_u = np.std(data["lower"][bi, mi], ddof=1), np.std(data["upper"][bi, mi], ddof=1)
            S = np.hypot(sd_l, sd_u)/np.sqrt(2)
            assert actual["S"][bi, mi] == S
            assert actual["Ase"][bi, mi] == S/.5
            assert actual["Apop"][bi, mi] == S*np.sqrt(20)
            assert actual["Vse"][bi, mi] == (S/.5)**2
        for key in ("lower_sd", "upper_sd", "S", "Ase", "Apop", "Vse"):
            assert actual[key][bi, 0] == 0


def test_invalid_local_method_B_is_na_not_noncoverage_counts_preserved():
    data = endpoint_fixture()
    data["valid"][0, 2, 1] = False
    actual = m06.original_metrics(data, .5)
    assert not actual["complete"][0, 2] and actual["complete"][1, 2]
    assert actual["total_inner"][0, 2] == 4
    assert actual["valid_inner"][0, 2] == 3
    assert actual["invalid_inner"][0, 2] == 1
    for key in ("lower_sd", "upper_sd", "S", "Ase", "Apop", "Vse", "C", "T", "mean_width", "warning_rate"):
        assert np.isnan(actual[key][0, 2])
    assert actual["complete"][:, :2].all()


@pytest.mark.parametrize("se", [0, -1, np.nan, np.inf, True, "0.5", 10**1000])
def test_invalid_se_retains_denominators_without_primary_metrics(se):
    actual = m06.original_metrics(endpoint_fixture(), se)
    assert not actual["complete"].any()
    assert np.isnan(actual["Ase"]).all() and np.isnan(actual["C"]).all()
    np.testing.assert_array_equal(actual["total_inner"], [[1, 4, 4]]*2)


def test_k1_is_rejected_here_and_accepted_m05_na_stays_unchanged():
    data = endpoint_fixture()
    with pytest.raises(ValueError, match="K>=2"):
        m06.original_metrics({k: v[:, :, :1] for k, v in data.items()}, .5)
    old = m05.original_metrics({k: v[0, :, 0] for k, v in data.items()})
    for key in ("lower_sd", "upper_sd", "S", "Ase", "Apop"):
        assert np.isnan(old[key]).all()


@pytest.mark.parametrize("key,index", [("lower", (0, 0, 1)), ("upper", (1, 0, 0)),
                                       ("valid", (0, 0, 1)), ("tail_warning", (1, 0, 2))])
def test_t_control_must_be_invariant_across_inner_and_B(key, index):
    data = endpoint_fixture()
    data[key][index] = not data[key][index] if data[key].dtype.kind == "b" else .8
    with pytest.raises(ValueError, match="invariant"):
        m06.original_metrics(data, .5)


def test_constant_bootstrap_endpoints_are_exact_zero_but_width_overflow_invalid():
    data = endpoint_fixture()
    data["lower"][:, 1] = .8
    data["upper"][:, 1] = 1.8
    actual = m06.original_metrics(data, .5)
    assert np.all(actual["lower_sd"][:, 1] == 0)
    assert np.all(actual["Vse"][:, 1] == 0)
    data["lower"][0, 2, 0], data["upper"][0, 2, 0] = -1e308, 1e308
    actual = m06.original_metrics(data, .5)
    assert not actual["complete"][0, 2] and actual["invalid_inner"][0, 2] == 1


@pytest.mark.parametrize("bounds,status,equivalent", [
    ((-.009, .009), "not_supported", True), ((.011, .02), "supported", False),
    ((-.02, -.011), "supported", False), ((-.01, .009), "inconclusive", False),
    ((-.009, .01), "inconclusive", False), ((.01, .02), "inconclusive", False),
    ((-.02, -.01), "inconclusive", False), ((0, 0), "inconclusive", False),
    ((np.nan, .009), "inconclusive", False)])
def test_h1_ase_units_strict_margin_and_no_seed_gate(bounds, status, equivalent):
    actual = m06.practical_difference_status(*bounds)
    assert actual["status"] == status and actual["equivalence_supported"] == equivalent
    assert m06.practical_difference_status(*bounds, complete=False)["status"] == "inconclusive"


@pytest.mark.parametrize("bounds,status", [((-.02, -.001), "supported"),
    ((0, .01), "not_supported"), ((.001, .01), "not_supported"),
    ((-.01, 0), "inconclusive"), ((-.01, .01), "inconclusive"),
    ((0, 0), "inconclusive"), ((np.nan, .01), "inconclusive")])
def test_h2_boundaries_and_missing(bounds, status):
    assert m06.reduction_status(*bounds)["status"] == status
    assert m06.reduction_status(*bounds, complete=False)["status"] == "inconclusive"


def test_h2_joint_status_contrary_has_priority_and_cluster_intervals():
    assert m06.combine_reduction_status(["supported", "supported"]) == "supported"
    assert m06.combine_reduction_status(["not_supported", "inconclusive"]) == "not_supported"
    assert m06.combine_reduction_status(["supported", "inconclusive"]) == "inconclusive"
    with pytest.raises(ValueError):
        m06.combine_reduction_status(["supported"])
    assert not m06.mean_t_interval(np.full(200, .8))["valid"]
    assert not m06.mean_t_interval([1, np.nan, 2])["valid"]
    assert not m06.welch_interval([.8]*4, [.8]*4)["valid"]
    one, zero = np.array([.1, .2, .4, .8]), np.array([.1, .3, .5, .9])
    actual = m06.welch_interval(one, zero)
    reference = stats.ttest_ind(one, zero, equal_var=False).confidence_interval(.99)
    assert [actual["low"], actual["high"]] == pytest.approx([reference.low, reference.high])
    assert actual["N1"] == actual["N0"] == 4


@pytest.fixture(scope="module")
def fixed_blobs():
    """Synthetic saved files with deterministic values; no original RNG."""
    design = {"distributions": list(m06.DISTRIBUTIONS), "n": 20, "master_seeds": [0, 1, 2],
              "R": 10000, "K": 1, "B": 4999, "alpha": .05, "methods": list(m06.METHODS)}
    blobs = {m06.M05_MANIFEST_PATH: json.dumps({"question_id": "RQ-M05", "design": design}).encode()}
    for distribution in m06.DISTRIBUTIONS:
        for seed in (0, 1, 2):
            samples = np.arange(10000, dtype=float)[:, None]/10000 + np.arange(20)[None, :]/20 + seed + m06.DISTRIBUTION_IDS[distribution]
            hashes = np.asarray([m06.sample_sha256(row).encode() for row in samples], dtype="S64")
            stream = BytesIO()
            np.savez_compressed(stream, samples=samples, outer_id=np.arange(10000, dtype=np.uint16),
                sample_sha256=hashes, distribution=distribution, master_seed=seed,
                n=20, R=10000, observed_original_count=10000, K=1, B=4999,
                alpha=.05, namespace=0x4D3035, methods=np.asarray(m06.METHODS))
            blobs[f"evidence/rq_m05/panels/{distribution}_seed{seed}.npz"] = stream.getvalue()
    return blobs


def test_verified_blob_compares_local_full_bytes_with_accepted_git_blob(monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"accepted")
    monkeypatch.setattr(m06, "_accepted_blob", lambda root, path: b"accepted")
    assert m06._verified_blob(Path("."), "fixture.npz") == b"accepted"
    monkeypatch.setattr(m06, "_accepted_blob", lambda root, path: b"different")
    with pytest.raises(ValueError, match="differs from accepted"):
        m06._verified_blob(Path("."), "fixture.npz")


def test_input_manifest_fixed_ids_hashes_readonly_no_rng_or_endpoint_selection(monkeypatch, fixed_blobs):
    seen = []
    def blob(root, path):
        seen.append(path)
        return fixed_blobs[path]
    monkeypatch.setattr(m06, "_verified_blob", blob)
    monkeypatch.setattr(m06, "rng_for_panel", lambda *a, **kw: pytest.fail("original RNG requested"))
    inputs = m06.make_inputs_manifest()
    assert inputs["accepted_head"] == m06.ACCEPTED_HEAD
    assert len(inputs["source_files_sha256"]) == 7
    assert inputs["selected_originals"] == 1200
    assert set(seen) == set(m06.INPUT_PATHS)
    assert "/Users/" not in json.dumps(inputs)
    cells = m06.load_inputs(inputs)
    m06.validate_inputs_manifest(inputs)
    assert len(cells) == 6
    for selection in inputs["selections"]:
        cell = cells[(selection["distribution"], selection["master_seed"])]
        assert cell["samples"].shape == (200, 20)
        np.testing.assert_array_equal(cell["outer_id"], np.arange(200))
        assert selection["outer_ids"] == list(range(200))
        assert selection["selected_samples_sha256"] == m06.sample_sha256(cell["samples"])
        for array in cell.values():
            assert not array.flags.writeable
        assert selection["sample_sha256"] == [m06.sample_sha256(row) for row in cell["samples"]]
    assert inputs == m06.make_inputs_manifest()
    changed = deepcopy(inputs)
    changed["selections"][0]["outer_ids"][-1] = 200
    for operation in (m06.validate_inputs_manifest, m06.load_inputs):
        with pytest.raises(ValueError, match="does not match"):
            operation(changed)


@pytest.mark.parametrize("drift", ["hash", "ids", "metadata", "manifest"])
def test_saved_input_drift_stops_without_repair_or_regeneration(monkeypatch, fixed_blobs, drift):
    blobs = dict(fixed_blobs)
    path = "evidence/rq_m05/panels/gamma_seed0.npz"
    if drift == "manifest":
        raw = json.loads(blobs[m06.M05_MANIFEST_PATH])
        raw["design"]["K"] = 30
        blobs[m06.M05_MANIFEST_PATH] = json.dumps(raw).encode()
    else:
        with np.load(BytesIO(blobs[path]), allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        if drift == "hash":
            arrays["samples"][9999, 0] += 1  # verify even unselected original hashes
        elif drift == "ids":
            arrays["outer_id"][[0, 1]] = arrays["outer_id"][[1, 0]]
        else:
            arrays["namespace"] = np.array(m06.MAIN_NAMESPACE)
        stream = BytesIO()
        np.savez_compressed(stream, **arrays)
        blobs[path] = stream.getvalue()
    monkeypatch.setattr(m06, "_verified_blob", lambda root, path: blobs[path])
    monkeypatch.setattr(m06, "rng_for_panel", lambda *a, **kw: pytest.fail("repair RNG requested"))
    with pytest.raises(ValueError, match="M05"):
        m06.make_inputs_manifest()


def test_real_accepted_inputs_integration_is_read_only():
    # Verifies the accepted Git blobs and 60k hashes, selecting existing1200 only.
    manifest = m06.make_inputs_manifest()
    cells = m06.load_inputs(manifest)
    assert manifest["source_files_sha256"][m06.M05_MANIFEST_PATH] == "d02a4b1d2ea1f26d96549e35014903c0c86df32dc8b8507bf81cc2f6dfdc8454"
    assert sum(len(cell["samples"]) for cell in cells.values()) == 1200
    assert all(np.array_equal(cell["outer_id"], np.arange(200)) for cell in cells.values())
    assert all(sha256(cell["samples"].astype("<f8").tobytes()).hexdigest() in
               [s["selected_samples_sha256"] for s in manifest["selections"]] for cell in cells.values())
