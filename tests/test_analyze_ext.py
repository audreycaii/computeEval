import json
import math
import os

import numpy as np

from src import analyze as A
from src import analyze_ext as X

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "runs_small.jsonl")


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def _rows(**over):
    rows = [json.loads(l) for l in open(FIX)]
    for r in rows:
        r.update(over)
    return rows


def test_effort_comparison_has_both_levels(tmp_path):
    high = _write(tmp_path, "high.jsonl", _rows(effort="high"))
    out = X.effort_comparison(FIX, high, n_boot=30)
    assert set(out["effort"]) == {"medium", "high"}
    assert set(out["model"]) == {"mA", "mB"}
    row = out[(out.model == "mA") & (out.effort == "high")].iloc[0]
    assert 0.0 <= row["N"] <= 1.0 and row["N_lo"] <= row["N"] <= row["N_hi"]


def test_provenance_uses_main_clean_runs_as_reference(tmp_path):
    rows = [r for r in _rows() if r["condition"] == "C1"]
    files = {}
    for style, shift in (("filing", 1), ("wire", 0), ("forum", 0)):
        rs = [dict(r) for r in rows]
        for r in rs:
            r["plant_style"] = style
            if r["parsed"] and shift:
                r["parsed"] = dict(r["parsed"], rating=min(5, r["parsed"]["rating"] + shift))
        files[style] = _write(tmp_path, style + ".jsonl", rs)
    cells, pooled = X.provenance(FIX, files, n_boot=30)
    assert set(cells["style"]) == {"filing", "wire", "forum"}
    a = cells[(cells.model == "mA") & (cells.ticker == "NBIS")].set_index("style")
    assert a.loc["filing", "E1"] > a.loc["wire", "E1"]  # shifted ratings flip more
    assert "E1_lo" in pooled.columns and set(pooled["style"]) == {"filing", "wire", "forum"}


def test_repeats_needed_shrinks_with_n():
    out = X.repeats_needed(FIX, ns=(3, 5, 10), n_draw=40, seed=0)
    assert list(out.columns) >= ["model", "n", "E1_halfwidth", "E1_sign_agreement"]
    m = out[out.model == "mA"].set_index("n")
    assert m.loc[3, "E1_halfwidth"] >= m.loc[10, "E1_halfwidth"]
    assert 0.0 <= m.loc[5, "E1_sign_agreement"] <= 1.0


def test_write_ext_numbers_macros(tmp_path):
    high = _write(tmp_path, "high.jsonl", _rows(effort="high"))
    eff = X.effort_comparison(FIX, high, n_boot=20)
    rows = [r for r in _rows() if r["condition"] == "C1"]
    files = {s: _write(tmp_path, s + ".jsonl", [dict(r, plant_style=s) for r in rows]) for s in ("filing", "wire", "forum")}
    cells, pooled = X.provenance(FIX, files, n_boot=20)
    rep = X.repeats_needed(FIX, ns=(3, 5), n_draw=20)
    path = str(tmp_path / "numbers_ext.tex")
    X.write_ext_numbers(eff, pooled, rep, path)
    tex = open(path).read()
    assert "\\newcommand{\\xmAHighNoise}{" in tex
    assert "\\newcommand{\\xmAFilingEone}{" in tex
    assert "\\newcommand{\\xmARepFiveHalfwidth}{" in tex
    import re
    names = re.findall(r"\\newcommand\{\\([^}]+)\}", tex)
    assert all(n.isalpha() for n in names)
