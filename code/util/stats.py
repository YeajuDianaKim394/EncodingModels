import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from util.atlas import get_brainmask


def ttest_1samp(
    values: np.ndarray,
    popmean: int = 0,
    correlations: bool = True,
    alternative: str = "two-sided",
    alpha: float = 0.01,
    method="bonf",
):
    if correlations:
        values = np.arctanh(values)

    fgmask = get_brainmask()

    ttest = stats.ttest_1samp(values, popmean=popmean, alternative=alternative)

    pvalues = ttest.pvalue
    multiple = multipletests(pvalues[fgmask], alpha=alpha, method=method)

    reject = np.zeros_like(pvalues, dtype=bool)
    reject[*fgmask.nonzero()] = multiple[0]

    return reject


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
