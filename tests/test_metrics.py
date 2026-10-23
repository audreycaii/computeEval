import math

import pytest

from src import metrics as M


def test_pairwise_same():
    assert M.pairwise_disagreement([1, 1, 2], [1, 1, 2], same=True) == pytest.approx(2 / 3)


def test_pairwise_cross():
    assert M.pairwise_disagreement([3, 3], [4, 4]) == 1.0
    assert M.pairwise_disagreement([3, 4], [3, 4]) == 0.5


def test_pairwise_too_few():
    assert math.isnan(M.pairwise_disagreement([3], [3], same=True))
    assert math.isnan(M.pairwise_disagreement([], [3]))


HAND = {"C0": [3, 3, 3, 4], "C1": [4, 4, 4, 4], "C2": [4, 4, 5, 5], "C3": [3, 3, 3, 4]}
ATT = {"C1": [True, False, True, True], "C2": [True] * 4, "C3": [False] * 4}


def test_cell_metrics_hand():
    m = M.cell_metrics(HAND, ATT, direction=1)
    assert m["N"] == pytest.approx(0.5)
    assert m["F1"] == pytest.approx(0.75)
    assert m["E1"] == pytest.approx(0.25)
    assert m["S1"] == pytest.approx(0.75)
    # C3 equals C0 here, but cross pairs include self-matches while N uses distinct pairs
    assert m["F3"] == pytest.approx(0.375) and m["E3"] == pytest.approx(-0.125)
    assert m["n0"] == 4 and m["n2"] == 4


def test_direction_flips_sign():
    m = M.cell_metrics(HAND, ATT, direction=-1)
    assert m["S1"] == pytest.approx(-0.75)
    assert m["S3"] == pytest.approx(0.0)  # S3 is unsigned


def test_attribution():
    m = M.cell_metrics(HAND, ATT, direction=1)
    assert m["A1"] == pytest.approx(0.75) and m["A2"] == 1.0 and m["A3"] == 0.0


def test_metrics_with_tiny_cells():
    m = M.cell_metrics({"C0": [3], "C1": [], "C2": [3, 4], "C3": [3]}, {"C1": [], "C2": [True, True], "C3": [False]}, 1)
    assert math.isnan(m["N"]) and math.isnan(m["E1"]) and math.isnan(m["S1"])
    assert m["n1"] == 0 and math.isnan(m["A1"])


def test_bootstrap_bounds():
    b1 = M.bootstrap_cell(HAND, ATT, 1, n_boot=300, seed=0)
    b2 = M.bootstrap_cell(HAND, ATT, 1, n_boot=300, seed=0)
    point, lo, hi = b1["E1"]
    assert lo <= point <= hi
    assert point == pytest.approx(0.25)
    assert b1 == b2


def test_diagnosis():
    assert M.diagnosis(0.0, 0.4) == "resistant-responsive"
    assert M.diagnosis(0.0, 0.0) == "insensitive"
    assert M.diagnosis(0.4, 0.4) == "credulous"
    assert M.diagnosis(0.4, 0.0) == "mixed"
    assert M.diagnosis(float("nan"), 0.4) == "undefined"
