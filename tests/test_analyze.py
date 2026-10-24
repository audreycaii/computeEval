import math
import os

from src import analyze as A

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "runs_small.jsonl")


def test_load_runs_shape():
    df = A.load_runs(FIX)
    assert len(df) == 45
    assert df["attributed"].dtype == bool
    assert set(df["model"]) == {"mA", "mB"}
    assert "fair_value_usd" in df.columns and "rating" in df.columns


def test_per_cell_has_expected_values():
    cells = A.per_cell(A.load_runs(FIX), n_boot=100)
    row = cells[(cells.model == "mA") & (cells.ticker == "NBIS")].iloc[0]
    assert abs(row["E1"]) < 1e-9
    assert row["E2"] > 0
    assert row["E1_lo"] <= row["E1"] <= row["E1_hi"]
    assert row["N"] == 0.0


def test_analyze_handles_missing_cell(tmp_path):
    out = str(tmp_path / "res")
    A.main(["--runs", FIX, "--out", out, "--n-boot", "50"])
    for name in ["metrics.csv", "summary.csv", "table1.tex", "fig1_ratings.pdf", "fig2_excess.pdf", "cost.txt"]:
        assert os.path.exists(os.path.join(out, name)), name
    cells = A.per_cell(A.load_runs(FIX), n_boot=50)
    row = cells[(cells.model == "mB") & (cells.ticker == "CRWV")].iloc[0]
    assert math.isnan(row["E3"]) and row["n3"] == 0


def test_pooled_has_one_row_per_model_and_diagnosis():
    cells = A.per_cell(A.load_runs(FIX), n_boot=50)
    pooled = A.pooled(cells)
    assert list(pooled["model"]) == ["mA", "mB"]
    assert set(pooled["diagnosis"]) <= {"resistant-responsive", "insensitive", "credulous", "mixed", "undefined"}


def test_table1_is_valid_latex(tmp_path):
    cells = A.per_cell(A.load_runs(FIX), n_boot=50)
    path = str(tmp_path / "t.tex")
    A.write_table1(A.pooled(cells), path)
    tex = open(path).read()
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert tex.count("\\\\") >= 3  # header + 2 model rows


def test_numbers_tex_macros(tmp_path):
    df = A.load_runs(FIX)
    cells = A.per_cell(df, n_boot=50)
    pooled_df = A.pooled(cells)
    all_rows = A.load_all_rows(FIX)
    path = str(tmp_path / "numbers.tex")
    A.write_numbers(pooled_df, cells, all_rows, path)
    tex = open(path).read()
    assert "\\newcommand{\\nValidRuns}{45}" in tex
    assert "\\newcommand{\\nTotalRuns}{48}" in tex
    assert "\\newcommand{\\mmAEone}{" in tex  # per-model macros: \m + letters of the model label
    assert "\\newcommand{\\mmADiag}{resistant-responsive}" in tex
    assert "\\newcommand{\\mmAEoneNet}{" in tex
    assert "\\newcommand{\\costTotal}{0.48}" in tex


def test_appendix_tables(tmp_path):
    cells = A.per_cell(A.load_runs(FIX), n_boot=50)
    pa = str(tmp_path / "tableA.tex")
    A.write_table_cells(cells, pa)
    tex = open(pa).read()
    assert "\\toprule" in tex and "NBIS" in tex and "CRWV" in tex and tex.count("\\\\") >= 5
    pp = str(tmp_path / "packets.tex")
    A.write_packet_table(pp)
    tex = open(pp).read()
    assert "NBIS-20" in tex and "NVDA-22" in tex and "synthetic" in tex.lower()
    assert "\\begin{aeslongtable}" in tex


def test_numbers_tex_macro_names_have_no_digits(tmp_path):
    import re
    df = A.load_runs(FIX)
    cells = A.per_cell(df, n_boot=20)
    path = str(tmp_path / "numbers.tex")
    A.write_numbers(A.pooled(cells), cells, A.load_all_rows(FIX), path)
    names = re.findall(r"\\newcommand\{\\([^}]+)\}", open(path).read())
    assert names and all(n.isalpha() for n in names), [n for n in names if not n.isalpha()][:5]


def test_num_formats_negative_zero_as_zero():
    assert A._num(-0.0) == "0.00"
    assert A._num(-0.004) == "0.00"
    assert A._num(-0.006) == "-0.01"
    assert A._fmt(-0.0, -0.0, 0.1).startswith("0.00")


def test_pooled_intervals_average_resampled_statistics():
    import numpy as np
    df = A.load_runs(FIX)
    cells = A.per_cell(df, n_boot=60)
    pooled = A.pooled(cells)
    row = pooled[pooled.model == "mA"].iloc[0]
    assert row["N_lo"] >= 0.0 and row["N_hi"] <= 1.0
    # the interval is the percentile of the mean over tickers of the per-ticker resampled N
    samples = cells.attrs["samples"]
    stacked = np.mean([samples[("mA", t)]["N"] for t in ("CRWV", "NBIS")], axis=0)
    assert row["N_lo"] == float(np.percentile(stacked, 2.5))


def test_num_rounds_half_up():
    assert A._num(0.475) == "0.48"
    assert A._num(0.4875) == "0.49"
    assert A._num(0.3375) == "0.34"


def test_load_runs_rejects_duplicate_keys(tmp_path):
    import json
    import pytest
    src = open(FIX).read().splitlines()
    p = tmp_path / "dup.jsonl"
    p.write_text("\n".join(src + [src[0]]) + "\n")
    with pytest.raises(ValueError):
        A.load_runs(str(p))


def test_unknown_ids_are_counted(tmp_path):
    import json
    rows = [json.loads(l) for l in open(FIX)]
    rows[0]["unknown_ids"] = ["NBIS-99"]
    p = tmp_path / "u.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    df = A.load_runs(str(p))
    cells = A.per_cell(df, n_boot=20)
    assert int(cells["n_unknown"].sum()) == 1
    out = str(tmp_path / "numbers.tex")
    A.write_numbers(A.pooled(cells), cells, A.load_all_rows(str(p)), out)
    assert "\\newcommand{\\nUnknownIds}{1}" in open(out).read()


def test_load_runs_defaults_variant_and_marker(tmp_path):
    import json
    rows = [json.loads(l) for l in open(FIX)]
    rows[0]["parsed"]["rationale"] = "The Synthetic item is discounted."
    rows[1]["parsed"]["drivers"][0]["claim"] = "A SYNTHETIC wire report says so"
    rows[2]["system_variant"] = "skeptical"
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    df = A.load_runs(str(p)).set_index("run_id")
    assert df.loc[rows[0]["run_id"], "mentions_synthetic"] == True
    assert df.loc[rows[1]["run_id"], "mentions_synthetic"] == True
    assert df["mentions_synthetic"].sum() == 2
    assert df.loc[rows[2]["run_id"], "system_variant"] == "skeptical"
    assert (df.drop(index=rows[2]["run_id"])["system_variant"] == "default").all()


def test_load_runs_marker_visible_defaults_true(tmp_path):
    import json
    rows = [json.loads(l) for l in open(FIX)]
    rows[0]["marker_visible"] = False
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    df = A.load_runs(str(p)).set_index("run_id")
    assert df.loc[rows[0]["run_id"], "marker_visible"] == False
    assert df.loc[rows[1]["run_id"], "marker_visible"] == True
    assert df["marker_visible"].dtype == bool
