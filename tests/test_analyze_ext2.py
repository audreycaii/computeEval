import itertools
import json
import os

import numpy as np
import pandas as pd

from src import analyze as A
from src import analyze_ext2 as Y

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "runs_small.jsonl")


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def _enumerated_median_pmf(p, k):
    out = np.zeros(5)
    for combo in itertools.product(range(5), repeat=k):
        out[int(np.median(combo))] += np.prod([p[c] for c in combo])
    return out


def test_median_pmf_matches_enumeration():
    rng = np.random.default_rng(1)
    for k in (1, 3, 5):
        p = rng.dirichlet(np.ones(5))
        assert np.allclose(Y.median_pmf(p, k), _enumerated_median_pmf(p, k), atol=1e-12)
        one_hot = np.array([0, 0, 1.0, 0, 0])
        assert np.allclose(Y.median_pmf(one_hot, k), one_hot)
    assert np.isclose(Y.median_pmf(np.array([0.5, 0, 0, 0, 0.5]), 3).sum(), 1.0)


def test_vote_noise_floor_falls_with_k():
    # one cell: noisy clean ratings, plant shifts them consistently by one notch
    rows = []
    for i, r in enumerate([2, 3, 3, 3, 4, 3, 2, 3, 3, 4]):
        rows.append({"run_id": "a%d" % i, "model": "mA", "ticker": "NBIS", "condition": "C0", "direction": 1, "repeat": i,
                     "special_item_id": None, "parse_ok": True, "unknown_ids": [], "input_tokens": 1, "output_tokens": 1, "cost_usd": 0,
                     "parsed": {"rating": r, "confidence": 0.5, "fair_value_usd": 100.0, "rationale": "x",
                                "drivers": [{"claim": "c", "evidence_ids": ["NBIS-04"]}] * 3}})
    for cond, shift, sid in (("C1", 1, "NBIS-21"), ("C2", 1, "NBIS-20")):
        for i, r in enumerate([3, 4, 4, 4, 5, 4, 3, 4, 4, 5]):
            rows.append({"run_id": "%s%d" % (cond, i), "model": "mA", "ticker": "NBIS", "condition": cond, "direction": 1, "repeat": i,
                         "special_item_id": sid, "parse_ok": True, "unknown_ids": [], "input_tokens": 1, "output_tokens": 1, "cost_usd": 0,
                         "parsed": {"rating": r, "confidence": 0.5, "fair_value_usd": 100.0, "rationale": "x",
                                    "drivers": [{"claim": "c", "evidence_ids": [sid]}] * 3}})
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    path = _write(d, "v.jsonl", rows)
    cells, pooled = Y.vote(path, ks=(1, 3, 5), n_boot=50)
    n = {int(r.k): r.N for r in cells.itertuples()}
    assert n[1] > n[3] > n[5]
    e1 = {int(r.k): r.E1 for r in cells.itertuples()}
    assert e1[5] >= e1[1] - 1e-9  # a consistent shift survives voting
    p0 = np.bincount([2, 3, 3, 3, 4, 3, 2, 3, 3, 4], minlength=6)[1:] / 10.0
    assert np.isclose(n[1], 1 - np.sum(p0 ** 2))
    assert set(pooled.columns) >= {"k", "model", "N", "N_lo", "N_hi", "E1", "E2", "Sel", "Sel_lo", "Sel_hi"}
    assert list(pooled["k"]) == [1, 3, 5]


def test_selectivity_is_e2_minus_e1_paired():
    cells = Y.add_selectivity(A.per_cell(A.load_runs(FIX), n_boot=60))
    assert np.allclose(cells["Sel"], cells["E2"] - cells["E1"], equal_nan=True)
    key = ("mA", "NBIS")
    s = cells.attrs["samples"][key]
    assert np.allclose(s["Sel"], s["E2"] - s["E1"], equal_nan=True)
    pooled = Y.pooled_with_sel(cells)
    row = pooled[pooled.model == "mA"].iloc[0]
    assert np.isclose(row["Sel"], row["E2"] - row["E1"])
    assert row["Sel_lo"] <= row["Sel"] <= row["Sel_hi"]


def test_setting_frame_tags_rows():
    cells, pooled = Y.setting_frame(FIX, "medium", n_boot=40)
    assert (cells["setting"] == "medium").all() and (pooled["setting"] == "medium").all()
    assert cells.columns[0] == "setting" and pooled.columns[0] == "setting"
    assert "Sel" in pooled.columns


def test_settings_table_dashes_missing_e3(tmp_path):
    _, med = Y.setting_frame(FIX, "medium", n_boot=30)
    skep = med.copy()
    skep["setting"] = "skeptical"
    skep[["E3", "E3_lo", "E3_hi"]] = np.nan
    both = pd.concat([med, skep], ignore_index=True)
    out = str(tmp_path / "t.tex")
    Y.write_settings_table(both, out)
    txt = open(out).read()
    assert txt.count("\\\\") >= 4 and "nan" not in txt.lower()
    assert "--" in txt and "$\\Sigma$" in txt
    Y.fig_plane(both, pd.concat([Y.setting_frame(FIX, "medium", n_boot=30)[0]] * 1), str(tmp_path / "plane.pdf"))
    assert os.path.getsize(str(tmp_path / "plane.pdf")) > 1000
    Y.write_vote_table(Y.vote(FIX, ks=(1, 3), n_boot=20)[1], str(tmp_path / "v.tex"))
    assert "\\toprule" in open(str(tmp_path / "v.tex")).read()


def test_combine_settings_concats_frames_with_sample_attrs():
    c1, p1 = Y.setting_frame(FIX, "medium", n_boot=20, seed=0)
    c2, p2 = Y.setting_frame(FIX, "high", n_boot=20, seed=1)
    cells, pooled = Y.combine_settings([(c1, p1), (c2, p2)])
    assert set(cells["setting"]) == {"medium", "high"} and len(pooled) == len(p1) + len(p2)
    assert "samples" in p1.attrs  # the inputs keep their samples for the paired comparison
