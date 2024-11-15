import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from util.atlas import get_brainmask


def ttest_1samp(
    values: np.ndarray,
    popmean: int = 0,
    correlations: bool = True,
    alternative: str = "two-sided",
):
    """One sample t-test if sample mean is different than `popmean` (default 0).
    values should be a 1d array of `n` values
    """
    if correlations:
        values = np.arctanh(values)

    ttest = stats.ttest_1samp(values, popmean=popmean, alternative=alternative)
    p_values = ttest.pvalue
    return p_values


def correct_multiple_tests(
    p_values: np.ndarray,
    method: str = "bonf",
    alpha: float = 0.05,
    ignore_median_mask: bool = True,
):

    fgmask = slice(None)
    if ignore_median_mask:
        fgmask = get_brainmask()

    outputs = multipletests(p_values[fgmask], alpha=alpha, method=method)

    reject = np.zeros_like(p_values, dtype=bool)
    if ignore_median_mask:
        reject[*fgmask.nonzero()] = outputs[0]
    else:
        reject = outputs[0]

    return reject


def bootstrap_pvalues(
    observed: np.ndarray, null_distriution: np.ndarray, alternative: str = "greater"
) -> np.ndarray:
    """
    observed should be an 1d array of `n` values
    null_distribution should be a 2d array of shape (n_samples, n)
    """
    n_dims = len(observed)
    p_values = np.zeros(n_dims)
    for i in range(n_dims):
        nulldist = null_distriution[:, i] - observed[i]
        p_values[i] = calculate_pvalues(observed[i], nulldist, alternative=alternative)
    return p_values


def bootstrap_distribution(
    sample: np.ndarray, n_perms: int = 10000, statistic_function=np.mean
) -> np.ndarray:
    n_samples, n_dims = sample.shape
    bootstrap_dist = np.zeros((n_perms, n_dims), dtype=np.float32)
    for i_perm in range(n_perms):
        sub_sample = np.random.choice(n_samples, size=n_samples, replace=True)
        score_sample = sample[sub_sample]
        sample_stat = statistic_function(score_sample, axis=0)
        bootstrap_dist[i_perm] = sample_stat
    return bootstrap_dist


def calculate_pvalues(
    observed: np.ndarray,
    null_distribution: np.ndarray,
    alternative: str = "two-sided",
    adjustment: int = 1,
) -> np.ndarray:
    """Calculate p-value
    See https://github.com/scipy/scipy/blob/v1.10.1/scipy/stats/_resampling.py#L1133-L1602
    """
    n_resamples = len(null_distribution)

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less, "greater": greater, "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return pvalues
