"""Round-three analysis (METRICS-EXT3.md): Extension G, the synthetic marker visible to the model
(runs/labelled/) against hidden (runs/), main setting, plus the construction check on the hidden runs."""
import argparse
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import analyze as A
from src import analyze_ext2 as Y
from src import metrics as M
from src import packets as P

KEYS = ["N", "E1", "E2", "E3", "A1", "Sel"]
KEY_WORD = {"N": "Noise", "E1": "Eone", "E2": "Etwo", "E3": "Ethree", "A1": "Aone", "Sel": "Sel"}
KEY_LABEL = {"N": "$N$", "E1": "$E_1$", "E2": "$E_2$", "E3": "$E_3$", "A1": "$A_1$", "Sel": "$\\Sigma$"}
EPS = 0.05


def _frame(path: str, marker_visible: bool, setting: str, n_boot: int, seed: int) -> pd.DataFrame:
    df = A.load_runs(path)
    if set(df["marker_visible"]) != {marker_visible}:
        raise ValueError("%s holds marker_visible=%s, expected %r" % (path, sorted(set(df["marker_visible"])), marker_visible))
    if set(df["system_variant"]) != {"default"}:
        raise ValueError("%s is not a default-prompt run" % path)
    _, pooled = Y.setting_frame(path, setting, n_boot=n_boot, seed=seed, expected_variant="default")
    return pooled


def _pooled_samples(pooled: pd.DataFrame, model: str, key: str) -> Optional[np.ndarray]:
    arrs = [v[key] for (m, _t), v in pooled.attrs["samples"].items() if m == model and key in v]
    return np.nanmean(np.vstack(arrs), axis=0) if arrs else None


def marker_comparison(visible_path: str, hidden_path: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per model, pooled over tickers: each metric with the marker visible and hidden, the difference
    hidden - visible, and the percentile interval of the resample-by-resample difference (both runs
    bootstrapped with the same seed, pooled as in METRICS.md)."""
    vis = _frame(visible_path, True, "visible", n_boot, seed)
    hid = _frame(hidden_path, False, "hidden", n_boot, seed)
    rows = []
    for model in Y._model_order(set(vis["model"]) & set(hid["model"])):
        a, b = vis[vis.model == model].iloc[0], hid[hid.model == model].iloc[0]
        row = {"model": model, "diag_vis": a["diagnosis"], "diag_hid": b["diagnosis"]}
        for key in KEYS:
            row[key + "_vis"], row[key + "_hid"] = float(a[key]), float(b[key])
            row["d" + key] = float(b[key]) - float(a[key])
            sa, sb = _pooled_samples(vis, model, key), _pooled_samples(hid, model, key)
            if sa is not None and sb is not None and not math.isnan(row["d" + key]):
                n = min(len(sa), len(sb))
                _, row["d" + key + "_lo"], row["d" + key + "_hi"] = M.summarize_samples(row["d" + key], sb[:n] - sa[:n])
            else:
                row["d" + key + "_lo"] = row["d" + key + "_hi"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def hidden_marker_share(path: str) -> float:
    """Construction check: share of valid hidden-marker runs mentioning 'synthetic'."""
    df = A.load_runs(path)
    return float(df["mentions_synthetic"].mean()) if len(df) else float("nan")


def expectations(comp: pd.DataFrame, hidden_marker_share: float, eps: float = EPS) -> Dict[str, object]:
    """The pre-registered expectations of METRICS-EXT3.md that this module can evaluate (1, 2, 4)."""
    not_weaker = int((comp["dE1"] >= -eps - 1e-9).sum())
    same_diag = int((comp["diag_vis"] == comp["diag_hid"]).sum())
    return {
        "E1_not_weaker_models": not_weaker, "E1_not_weaker": not_weaker == len(comp),
        "hidden_marker_share": hidden_marker_share, "hidden_marker_zero": hidden_marker_share == 0.0,
        "same_diagnosis_models": same_diag, "same_diagnosis": same_diag >= 2,
    }


def write_marker_comparison_table(comp: pd.DataFrame, path: str) -> None:
    """Rows: model x {visible, hidden, difference}; columns: the metrics of KEYS. The difference row carries
    the paired interval in script size."""
    cols = [k for k in KEYS if k != "A1"]  # A1 is 1.00 throughout; the text says so
    lines = ["\\begin{tabular}{@{}ll" + "r" * len(cols) + "@{}}", "\\toprule",
             "Model & Marker & " + " & ".join(KEY_LABEL[k] for k in cols) + " \\\\", "\\midrule"]
    for _, r in comp.iterrows():
        label = A.MODEL_LABEL.get(r["model"], r["model"])
        lines.append("%s & visible & %s \\\\" % (label, " & ".join(A._num(r[k + "_vis"]) for k in cols)))
        lines.append(" & hidden & %s \\\\" % " & ".join(A._num(r[k + "_hid"]) for k in cols))
        lines.append(" & difference & %s \\\\" % " & ".join(Y._fmt_compact(r["d" + k], r["d" + k + "_lo"], r["d" + k + "_hi"]) for k in cols))
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_ext3_numbers(comp: pd.DataFrame, exp: Dict[str, object], hidden_share: float, path: str,
                       skeptical_deltas: Optional[Dict[str, Dict[str, float]]] = None,
                       labelled_marker: Optional[pd.DataFrame] = None) -> None:
    """Macros with prefix z and no digits: \\z<Model><Metric>{Vis,Hid,Delta,DeltaLo,DeltaHi}, diagnoses,
    expectation outcomes, and the sceptical-prompt E1 change under each marker setting when given."""
    lines = ["% Generated by src/analyze_ext3.py; do not edit by hand."]

    def put(name, value):
        lines.append("\\newcommand{\\%s}{%s}" % (name, value))

    for _, r in comp.iterrows():
        m = A._macro_name(r["model"])
        for key in KEYS:
            w = KEY_WORD[key]
            put("z%s%sVis" % (m, w), A._num(r[key + "_vis"])), put("z%s%sHid" % (m, w), A._num(r[key + "_hid"]))
            put("z%s%sDelta" % (m, w), A._num(r["d" + key]))
            put("z%s%sDeltaLo" % (m, w), A._num(r["d" + key + "_lo"])), put("z%s%sDeltaHi" % (m, w), A._num(r["d" + key + "_hi"]))
        put("z%sDiagVis" % m, r["diag_vis"]), put("z%sDiagHid" % m, r["diag_hid"])
    put("zHiddenMarkerShare", A._num(hidden_share))
    put("zExpectationEoneNotWeaker", "%d" % exp["E1_not_weaker_models"])
    put("zExpectationSameDiag", "%d" % exp["same_diagnosis_models"])
    put("zModels", "%d" % len(comp))
    for setting, deltas in (skeptical_deltas or {}).items():
        for model, d in deltas.items():
            put("z%sSkepticalDeltaEone%s" % (A._macro_name(model), setting.capitalize()), A._num(d))
    if labelled_marker is not None:
        for _, r in labelled_marker.iterrows():
            put("z%sMarkerVis%s%s" % (A._macro_name(r["model"]), Y.SETTING_WORD[r["setting"]],
                                        Y.COND_WORD.get(r["condition"], r["condition"])), A._num(r["share"]))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def skeptical_delta_e1(main_path: str, skeptical_path: str, n_boot: int, seed: int = 0) -> Dict[str, float]:
    """Pooled E1' - E1 per model for one marker setting (expectation 3 of METRICS-EXT3.md)."""
    _, main_pooled = Y.setting_frame(main_path, "medium", n_boot=n_boot, seed=seed, expected_variant="default")
    _, skep_pooled = Y.setting_frame(skeptical_path, "skeptical", n_boot=n_boot, seed=seed, expected_variant="skeptical")
    comp = Y.skeptical_comparison(main_pooled, skep_pooled)
    return {r["model"]: float(r["dE1"]) for _, r in comp.iterrows()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default=os.path.join(P.ROOT, "runs", "runs.jsonl"))
    ap.add_argument("--visible", default=os.path.join(P.ROOT, "runs", "labelled", "runs.jsonl"))
    ap.add_argument("--hidden-skeptical", default=os.path.join(P.ROOT, "runs", "ext_skeptical.jsonl"))
    ap.add_argument("--visible-skeptical", default=os.path.join(P.ROOT, "runs", "labelled", "ext_skeptical.jsonl"))
    ap.add_argument("--out", default=os.path.join(P.ROOT, "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    comp = marker_comparison(args.visible, args.hidden, n_boot=args.n_boot)
    share = hidden_marker_share(args.hidden)
    exp = expectations(comp, share)
    deltas = {}
    for setting, main_path, skep_path in (("vis", args.visible, args.visible_skeptical), ("hid", args.hidden, args.hidden_skeptical)):
        if os.path.exists(skep_path):
            deltas[setting] = skeptical_delta_e1(main_path, skep_path, args.n_boot)
    comp.to_csv(os.path.join(args.out, "ext3_marker_comparison.csv"), index=False)
    write_marker_comparison_table(comp, os.path.join(args.out, "ext3_marker_table.tex"))
    labelled = {k: v for k, v in (("medium", args.visible), ("skeptical", args.visible_skeptical)) if os.path.exists(v)}
    marker = Y.marker_check(labelled)
    marker.to_csv(os.path.join(args.out, "ext3_marker_labelled.csv"), index=False)
    Y.write_marker_table(marker, os.path.join(args.out, "ext3_marker_table_labelled.tex"))
    write_ext3_numbers(comp, exp, share, os.path.join(args.out, "numbers_ext3.tex"), skeptical_deltas=deltas,
                       labelled_marker=marker)
    pd.set_option("display.width", 200)
    print(comp[["model", "diag_vis", "diag_hid"] + [c for k in KEYS for c in (k + "_vis", k + "_hid", "d" + k, "d" + k + "_lo", "d" + k + "_hi")]].round(3).T.to_string())
    print("hidden marker share:", share)
    print("expectations:", exp)
    print("sceptical dE1:", deltas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
