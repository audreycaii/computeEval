import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from src import analyze as A
from src import analyze_ext2 as Y
from src import analyze_ext3 as Z

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "runs_small.jsonl")


def _rows():
    return [json.loads(l) for l in open(FIX)]


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_marker_comparison_same_file_gives_zero_differences(tmp_path):
    vis = _write(tmp_path, "vis.jsonl", _rows())
    hid = _write(tmp_path, "hid.jsonl", [dict(r, marker_visible=False) for r in _rows()])
    comp = Z.marker_comparison(vis, hid, n_boot=20)
    assert set(comp["model"]) == set(A.load_runs(vis)["model"])
    for key in Z.KEYS:
        assert {key + "_vis", key + "_hid", "d" + key, "d" + key + "_lo", "d" + key + "_hi"} <= set(comp.columns)
        ok = comp["d" + key].notna()
        assert np.allclose(comp.loc[ok, "d" + key], 0) and np.allclose(comp.loc[ok, "d" + key + "_lo"], 0) and np.allclose(comp.loc[ok, "d" + key + "_hi"], 0)
        assert np.allclose(comp.loc[ok, "d" + key], comp.loc[ok, key + "_hid"] - comp.loc[ok, key + "_vis"])
    assert set(comp["diag_vis"]) == set(comp["diag_hid"])


def test_marker_comparison_refuses_wrong_marker_flags(tmp_path):
    vis = _write(tmp_path, "vis.jsonl", _rows())
    hid = _write(tmp_path, "hid.jsonl", [dict(r, marker_visible=False) for r in _rows()])
    with pytest.raises(ValueError):
        Z.marker_comparison(hid, vis, n_boot=5)  # swapped
    with pytest.raises(ValueError):
        Z.marker_comparison(vis, vis, n_boot=5)


def test_marker_comparison_detects_weaker_plant(tmp_path):
    def shaped(marker_visible, plant_rating):
        out = []
        for r in _rows():
            r = dict(r, marker_visible=marker_visible)
            if r["parsed"]:
                if r["condition"] == "C1":
                    r["parsed"] = dict(r["parsed"], rating=plant_rating)
                elif r["condition"] == "C0":
                    r["parsed"] = dict(r["parsed"], rating=3)
            out.append(r)
        return out
    vis = _write(tmp_path, "vis.jsonl", shaped(True, 5))   # the plant moves every run
    hid = _write(tmp_path, "hid.jsonl", shaped(False, 3))  # the plant moves none
    comp = Z.marker_comparison(vis, hid, n_boot=20)
    assert (comp["E1_vis"] > 0.05).all() and (comp["E1_hid"] <= 0.05).all()
    assert (comp["dE1"] < -0.05).all() and (comp["dE1_hi"] < 0).all()
    exp = Z.expectations(comp, hidden_marker_share=0.0)
    assert exp["E1_not_weaker_models"] == 0 and exp["E1_not_weaker"] is False
    assert exp["hidden_marker_zero"] is True


def test_hidden_marker_share(tmp_path):
    rows = [dict(r, marker_visible=False) for r in _rows()]
    rows[0]["parsed"]["rationale"] = "looks synthetic"
    hid = _write(tmp_path, "hid.jsonl", rows)
    share = Z.hidden_marker_share(hid)
    df = A.load_runs(hid)
    assert share == pytest.approx(df["mentions_synthetic"].mean()) and share > 0


def test_ext3_outputs(tmp_path):
    vis = _write(tmp_path, "vis.jsonl", _rows())
    hid = _write(tmp_path, "hid.jsonl", [dict(r, marker_visible=False) for r in _rows()])
    comp = Z.marker_comparison(vis, hid, n_boot=20)
    tex = str(tmp_path / "t.tex")
    Z.write_marker_comparison_table(comp, tex)
    body = open(tex).read()
    assert "\\begin{tabular}" in body and "hidden" in body and "visible" in body
    nums = str(tmp_path / "n.tex")
    Z.write_ext3_numbers(comp, Z.expectations(comp, 0.0), 0.0, nums, skeptical_deltas={"vis": {"claude-opus-5-5": -0.46}, "hid": {"claude-opus-5-5": -0.10}})
    lines = open(nums).read().splitlines()
    names = [re.match(r"\\newcommand\{\\(\w+)\}", l).group(1) for l in lines if l.startswith("\\newcommand")]
    assert names and all(n.startswith("z") for n in names)
    assert not any(ch.isdigit() for n in names for ch in n)
    assert "zOpusSkepticalDeltaEoneVis" in names and "zOpusSkepticalDeltaEoneHid" in names
    assert "zHiddenMarkerShare" in names and "zExpectationEoneNotWeaker" in names
    marker = Y.marker_check({"medium": vis})
    Z.write_ext3_numbers(comp, Z.expectations(comp, 0.0), 0.0, nums, labelled_marker=marker)
    body = open(nums).read()
    assert "\\newcommand{\\zmAMarkerVisMedCone}" in body


def test_expectation_one_boundary_and_three(tmp_path):
    vis = _write(tmp_path, "vis.jsonl", _rows())
    hid = _write(tmp_path, "hid.jsonl", [dict(r, marker_visible=False) for r in _rows()])
    comp = Z.marker_comparison(vis, hid, n_boot=10)
    comp["dE1"] = [-0.05000000000000002, -0.0501][: len(comp)] + [0.0] * (len(comp) - 2)
    exp = Z.expectations(comp, 0.0)
    assert exp["E1_not_weaker_models"] == len(comp) - 1  # exactly eps counts; beyond eps does not
    comp["diag_vis"], comp["diag_hid"] = "mixed", "credulous"
    assert Z.expectations(comp, 0.0)["same_diagnosis_models"] == 0
    deltas = {"vis": {"claude-opus-5-5": -0.46}, "hid": {"claude-opus-5-5": -0.30}}
    assert Z.expectations(comp, 0.0, skeptical_deltas=deltas)["skeptical_dE1_smaller"] is True
    deltas["hid"]["claude-opus-5-5"] = -0.50
    assert Z.expectations(comp, 0.0, skeptical_deltas=deltas)["skeptical_dE1_smaller"] is False
    assert Z.expectations(comp, 0.0)["skeptical_dE1_smaller"] is None


def test_hidden_marker_share_pools_files_and_refuses_visible(tmp_path):
    rows = [dict(r, marker_visible=False) for r in _rows()]
    a = _write(tmp_path, "a.jsonl", rows)
    rows2 = [dict(r, marker_visible=False, run_id=r["run_id"] + "b") for r in _rows()]
    rows2[0]["parsed"]["rationale"] = "synthetic"
    b = _write(tmp_path, "b.jsonl", rows2)
    n = len(A.load_runs(a))
    assert Z.hidden_marker_share([a, b]) == pytest.approx(1 / (2 * n))
    with pytest.raises(ValueError):
        Z.hidden_marker_share([a, _write(tmp_path, "v.jsonl", _rows())])


def test_table_keeps_a1_when_not_saturated(tmp_path):
    vis = _write(tmp_path, "vis.jsonl", _rows())
    hid = _write(tmp_path, "hid.jsonl", [dict(r, marker_visible=False) for r in _rows()])
    comp = Z.marker_comparison(vis, hid, n_boot=10)
    tex = str(tmp_path / "t.tex")
    comp2 = comp.copy(); comp2["A1_vis"], comp2["A1_hid"] = 1.0, 1.0
    Z.write_marker_comparison_table(comp2, tex)
    assert "$A_1$" not in open(tex).read()
    comp2["A1_hid"] = 0.5
    Z.write_marker_comparison_table(comp2, tex)
    assert "$A_1$" in open(tex).read()
