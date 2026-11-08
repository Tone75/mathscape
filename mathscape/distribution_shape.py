"""RQ-M04 standardized populations and original-sample evaluation metrics.

Changing gamma shape changes skewness, kurtosis and support together. It is
not an intervention on skewness alone. Population standardization uses the
theoretical shape, never a sample's realized mean or variance. Existing mean
interval routines are reused unchanged and receive no population parameters.
"""

from numbers import Integral

import numpy as np
from scipy import stats

from mathscape import stability

MAIN_NAMESPACE = 0x4D3034
SMOKE_NAMESPACE = 0x534D3034
DISTRIBUTIONS = ("normal", "gamma_0.5", "gamma_1", "gamma_2", "gamma_4", "gamma_16")
DISTRIBUTION_IDS = {name: i + 1 for i, name in enumerate(DISTRIBUTIONS)}
GAMMA_SHAPES = {"gamma_0.5": .5, "gamma_1": 1., "gamma_2": 2., "gamma_4": 4., "gamma_16": 16.}
METHODS = ("t", "percentile", "bca")
B_VALUES = (999, 4999)
N = 20
MAX_OUTER = 400
MAX_INNER = 10
MAX_ID = 2**32 - 1
MAX_ELEMENTS = 1_000_000

# Public RQ-M03 numerical helpers remain unchanged, including the 4,999 cap.
prepare_sample = stability.prepare_sample
bootstrap_means = stability.bootstrap_means
mean_t_interval = stability.mean_t_interval
welch_interval = stability.welch_interval


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _distribution(distribution):
    if not isinstance(distribution, str) or distribution not in DISTRIBUTION_IDS:
        raise ValueError("distribution must be normal or a fixed standardized gamma population")
    return distribution


def population_info(distribution):
    """Theoretical generation/plotting metadata, never inference inputs."""
    distribution = _distribution(distribution)
    if distribution == "normal":
        return {"shape": None, "mean": 0., "variance": 1., "skewness": 0.,
                "excess_kurtosis": 0., "lower_bound": -np.inf}
    shape = GAMMA_SHAPES[distribution]
    return {"shape": shape, "mean": 0., "variance": 1., "skewness": 2 / np.sqrt(shape),
            "excess_kurtosis": 6 / shape, "lower_bound": -np.sqrt(shape)}


def population_distribution(distribution):
    """SciPy population distribution for reference checks and density plots.

    The shifted/scaled gamma has loc=-sqrt(r), scale=1/sqrt(r), shape=r.
    Singular support endpoints remain infinite densities; callers must omit
    them from finite display grids rather than replacing them with finite peaks.
    """
    distribution = _distribution(distribution)
    if distribution == "normal":
        return stats.norm(loc=0, scale=1)
    shape = GAMMA_SHAPES[distribution]
    return stats.gamma(a=shape, loc=-np.sqrt(shape), scale=1 / np.sqrt(shape))


def rng_for_panel(
    distribution, n=20, master_seed=0, outer_id=0, purpose_id=0, inner_id=0,
    namespace=MAIN_NAMESPACE,
):
    """M04 PCG64 key: (namespace,distribution ID,20,outer,purpose,inner).

    Master seed is the SeedSequence entropy. Original purpose=0/inner=0;
    bootstrap purpose=1/inner=0..9. Outer IDs are 0..399. B, alpha, method,
    display order and batching do not select streams. Different populations
    have independent keys even when their seed and outer IDs are equal.
    """
    distribution = _distribution(distribution)
    n = _integer(n, "n", N, N)
    master_seed = _integer(master_seed, "master_seed", 0, MAX_ID)
    outer_id = _integer(outer_id, "outer_id", 0, MAX_OUTER - 1)
    purpose_id = _integer(purpose_id, "purpose_id", 0, 1)
    inner_id = _integer(inner_id, "inner_id", 0, MAX_INNER - 1)
    namespace = _integer(namespace, "namespace", 0, MAX_ID)
    if purpose_id == 0 and inner_id != 0:
        raise ValueError("the original-sample stream must use inner_id=0")
    sequence = np.random.SeedSequence(entropy=master_seed,
        spawn_key=(namespace, DISTRIBUTION_IDS[distribution], n, outer_id, purpose_id, inner_id))
    return np.random.Generator(np.random.PCG64(sequence))


def generate_sample(distribution, n=20, master_seed=0, outer_id=0, namespace=MAIN_NAMESPACE):
    """Generate X=(Gamma(r,scale=1)-r)/sqrt(r), or the separate normal control."""
    rng = rng_for_panel(distribution, n, master_seed, outer_id, 0, 0, namespace)
    if distribution == "normal":
        sample = rng.standard_normal(n)
    else:
        shape = GAMMA_SHAPES[distribution]
        sample = (rng.gamma(shape=shape, scale=1.0, size=n) - shape) / np.sqrt(shape)
    if not np.isfinite(sample).all():
        raise ValueError("original_sample_nonfinite")
    return sample


def generate_bootstrap_means(
    sample, distribution, n=20, master_seed=0, outer_id=0, inner_id=0, Bmax=4999,
    namespace=MAIN_NAMESPACE, max_elements=MAX_ELEMENTS,
):
    rng = rng_for_panel(distribution, n, master_seed, outer_id, 1, inner_id, namespace)
    raw = np.asarray(sample)
    if raw.ndim != 1 or raw.size != n:
        raise ValueError("sample length and n disagree")
    return bootstrap_means(sample, rng, Bmax=Bmax, max_elements=max_elements)


def infer_prefixes(context, means, B_values=B_VALUES):
    """Use unchanged t/percentile/BCa definitions with M04's two default B's."""
    return stability.infer_prefixes(context, means, B_values=B_values)


def original_metrics(data, se, n=20):
    """Evaluate one original's (B,method,K) endpoints; no new mean intervals.

    Mbelow is U<0 and Mabove is L>0; T=Mbelow-Mabove. S is the raw pooled
    endpoint SD, Ase=S/se, Apop=S/(1/sqrt(20)). The known population SE enters
    this evaluation only. It is never substituted for the sample SE in t or
    bootstrap inference. Any invalid required inner interval leaves all primary
    metrics NA, preserving counts. t uses one control interval, not K copies.
    """
    _integer(n, "n", N, N)
    old = stability.original_metrics(data, se)
    output = {key: old[key] for key in ("complete", "total_inner", "valid_inner", "invalid_inner",
                                      "lower_sd", "upper_sd", "C", "mean_width", "warning_rate")}
    output["Mbelow"] = old["miss_below"]
    output["Mabove"] = old["miss_above"]
    output["T"] = output["Mbelow"] - output["Mabove"]
    output["S"] = np.hypot(old["lower_sd"], old["upper_sd"]) / np.sqrt(2)
    output["Ase"] = old["A"]
    output["Apop"] = output["S"] * np.sqrt(N)
    return output


def quantity_status(low, high, threshold, direction, seed_points, seed_threshold=0.0):
    """Strict pooled practical-effect gate and a separate seed-direction gate.

    M04 H1 uses pooled threshold=-.02 but seed_threshold=0, and H2 uses
    pooled threshold=.03 but seed_threshold=0. The caller must first enforce
    completeness for the particular required populations/method/B; unrelated
    invalid data must not poison an otherwise complete confirmatory comparison.
    A valid interval wholly opposite its pooled threshold has priority over
    unmet seed directions. Touching a threshold is always inconclusive.
    """
    if direction not in ("less", "greater"):
        raise ValueError("direction must be less or greater")
    if not np.isfinite([low, high, threshold, seed_threshold]).all() or low > high:
        return "inconclusive"
    contrary = low > threshold if direction == "less" else high < threshold
    if contrary:
        return "not_supported"
    if len(seed_points) != 3 or not np.isfinite(seed_points).all():
        return "inconclusive"
    desired = (high < threshold and all(point < seed_threshold for point in seed_points)
               if direction == "less" else low > threshold and all(point > seed_threshold for point in seed_points))
    return "supported" if desired else "inconclusive"
