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
    if not a or not b:
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


def bootstrap_cell(ratings: Dict[str, List[int]], attributed: Dict[str, List[bool]], direction: int,
                   n_boot: int = 2000, seed: int = 0) -> Dict[str, Tuple[float, float, float]]:
    """Point estimate and percentile 95% interval per metric, resampling runs within each condition."""
    point = cell_metrics(ratings, attributed, direction)
    rng = np.random.default_rng(seed)
    keys = [k for k in point if not k.startswith("n")]
    samples = {k: [] for k in keys}
    conds = ["C0"] + list(CONDS)
    for _ in range(n_boot):
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
            samples[k].append(m[k])
    out = {}
    for k in keys:
        s = np.array([v for v in samples[k] if not math.isnan(v)])
        if math.isnan(point[k]) or len(s) == 0:
            out[k] = (point[k], NAN, NAN)
        else:
            out[k] = (point[k], float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))
    return out


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
