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
