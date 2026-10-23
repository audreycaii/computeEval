"""Pre-registered metrics (see METRICS.md): pairwise disagreement, noise floor, excess flip,
mean shift, attribution, bootstrap intervals and the per-model diagnosis."""
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

CONDS = ("C1", "C2", "C3")
NAN = float("nan")


def pairwise_disagreement(a: Sequence[int], b: Sequence[int], same: bool = False) -> float:
    a, b = list(a), list(b)
    if same:
        n = len(a)
        if n < 2:
            return NAN
        diff = sum(1 for i in range(n) for j in range(i + 1, n) if a[i] != a[j])
        return diff / (n * (n - 1) / 2)
    if len(a) * len(b) < 2:
        return NAN
    diff = sum(1 for x in a for y in b if x != y)
    return diff / (len(a) * len(b))


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else NAN


def cell_metrics(ratings: Dict[str, List[int]], attributed: Dict[str, List[bool]], direction: int) -> Dict[str, float]:
    r0 = ratings.get("C0", [])
    out = {"N": pairwise_disagreement(r0, r0, same=True), "n0": len(r0)}
    for k in CONDS:
        rk = ratings.get(k, [])
        i = k[1]
        out["n" + i] = len(rk)
        out["F" + i] = pairwise_disagreement(rk, r0)
        out["E" + i] = out["F" + i] - out["N"]
        shift = _mean(rk) - _mean(r0)
        out["S" + i] = shift * direction if k in ("C1", "C2") else shift
        out["A" + i] = _mean([1.0 if a else 0.0 for a in attributed.get(k, [])])
    return out


def bootstrap_samples(ratings: Dict[str, List[int]], attributed: Dict[str, List[bool]], direction: int,
                      n_boot: int = 2000, seed: int = 0) -> Dict[str, np.ndarray]:
    """Resampled metric values, one array of length n_boot per metric, resampling runs within each
    condition. NaN where a resample cannot define the metric."""
    point = cell_metrics(ratings, attributed, direction)
    rng = np.random.default_rng(seed)
    keys = [k for k in point if not k.startswith("n")]
    samples = {k: np.full(n_boot, NAN) for k in keys}
    conds = ["C0"] + list(CONDS)
    for b in range(n_boot):
        rr, aa = {}, {}
        for c in conds:
            r = ratings.get(c, [])
            a = attributed.get(c, [])
            if r:
                idx = rng.integers(0, len(r), size=len(r))
                rr[c] = [r[i] for i in idx]
                aa[c] = [a[i] for i in idx] if len(a) == len(r) else []
            else:
                rr[c], aa[c] = [], []
        m = cell_metrics(rr, aa, direction)
        for k in keys:
            samples[k][b] = m[k]
    return samples


def summarize_samples(point: float, samples: np.ndarray) -> Tuple[float, float, float]:
    """Point estimate with the percentile 95% interval of its bootstrap samples."""
    s = samples[~np.isnan(samples)]
    if math.isnan(point) or len(s) == 0:
        return (point, NAN, NAN)
    return (point, float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


def bootstrap_cell(ratings: Dict[str, List[int]], attributed: Dict[str, List[bool]], direction: int,
                   n_boot: int = 2000, seed: int = 0) -> Dict[str, Tuple[float, float, float]]:
    """Point estimate and percentile 95% interval per metric (see bootstrap_samples)."""
    point = cell_metrics(ratings, attributed, direction)
    samples = bootstrap_samples(ratings, attributed, direction, n_boot=n_boot, seed=seed)
    return {k: summarize_samples(point[k], samples[k]) for k in samples}


def diagnosis(E1: float, E2: float, eps: float = 0.05) -> str:
    if math.isnan(E1) or math.isnan(E2):
        return "undefined"
    if E1 <= eps and E2 > eps:
        return "resistant-responsive"
    if E1 <= eps and E2 <= eps:
        return "insensitive"
    if E1 > eps and abs(E1 - E2) <= eps:
        return "credulous"
    return "mixed"
