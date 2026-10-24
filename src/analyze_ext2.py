"""Round-two extension analyses (METRICS-EXT2.md): selectivity and the (E1, E2) plane (D),
repeat-and-vote aggregation (E), the sceptical-prompt run and marker check (F)."""
import argparse
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

from src import analyze as A
from src import metrics as M
from src import packets as P

SETTINGS = ["medium", "high", "skeptical"]
SETTING_LABEL = {"medium": "medium", "high": "high", "skeptical": "sceptical prompt"}
SETTING_WORD = {"medium": "Med", "high": "High", "skeptical": "Skeptical"}
K_WORD = {1: "One", 3: "Three", 5: "Five"}
COND_WORD = {"C0": "Czero", "C1": "Cone", "C2": "Ctwo"}
RATINGS = np.arange(1, 6)


def _model_order(models: Iterable[str]) -> List[str]:
    return sorted(set(models), key=lambda m: list(A.MODEL_LABEL).index(m) if m in A.MODEL_LABEL else 99)


# ---------------------------------------------------------------- Extension D: selectivity

def add_selectivity(cells: pd.DataFrame) -> pd.DataFrame:
    """Sel = E2 - E1 per cell, with the interval taken from the paired bootstrap resamples that
    analyze.per_cell already stored in attrs["samples"]."""
    cells = cells.copy()
    samples = dict(cells.attrs.get("samples", {}))
    sel, lo, hi = [], [], []
    for r in cells.itertuples(index=False):
        key = (r.model, r.ticker)
        point = r.E2 - r.E1
        s = samples.get(key)
        if s is not None:
            arr = s["E2"] - s["E1"]
            s = dict(s)
            s["Sel"] = arr
            samples[key] = s
            point, l, h = M.summarize_samples(point, arr)
        else:
            l = h = float("nan")
        sel.append(point), lo.append(l), hi.append(h)
    cells["Sel"], cells["Sel_lo"], cells["Sel_hi"] = sel, lo, hi
    cells.attrs["samples"] = samples
    return cells


def pooled_with_sel(cells: pd.DataFrame) -> pd.DataFrame:
    """analyze.pooled plus the pooled selectivity, averaging ticker-level resamples as METRICS.md does."""
    pooled = A.pooled(cells)
    samples = cells.attrs.get("samples", {})
    sel, lo, hi = [], [], []
    for model in pooled["model"]:
        g = cells[cells.model == model]
        point = float(g["Sel"].astype(float).mean())
        arrs = [samples[(model, t)]["Sel"] for t in g["ticker"] if (model, t) in samples and "Sel" in samples[(model, t)]]
        if arrs:
            point, l, h = M.summarize_samples(point, np.nanmean(np.vstack(arrs), axis=0))
        else:
            l = h = float("nan")
        sel.append(point), lo.append(l), hi.append(h)
    pooled["Sel"], pooled["Sel_lo"], pooled["Sel_hi"] = sel, lo, hi
    return pooled


def setting_frame(path: str, setting: str, n_boot: int = 2000, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell and pooled metrics (with Sel) for one run file, tagged with its setting."""
    cells = add_selectivity(A.per_cell(A.load_runs(path), n_boot=n_boot, seed=seed))
    pooled = pooled_with_sel(cells)
    samples = cells.attrs["samples"]
    cells.insert(0, "setting", setting)
    pooled.insert(0, "setting", setting)
    cells.attrs["samples"] = samples
    pooled.attrs["samples"] = samples
    return cells, pooled


def combine_settings(frames: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate (cells, pooled) pairs from several settings. The inputs keep their bootstrap samples in
    attrs; the combined frames carry none, because pandas cannot compare array-valued attrs on concat."""
    def bare(df):
        out = df.copy()
        out.attrs = {}
        return out
    cells = pd.concat([bare(c) for c, _ in frames], ignore_index=True)
    pooled = pd.concat([bare(p) for _, p in frames], ignore_index=True)
    return cells, pooled


# ---------------------------------------------------------------- Extension E: repeat-and-vote

def median_pmf(p: np.ndarray, k: int) -> np.ndarray:
    """Exact distribution of the median of k i.i.d. draws from pmf p over the five ratings.
    P(median <= v) = P(Bin(k, F(v)) >= (k + 1) / 2) for odd k."""
    if k % 2 == 0:
        raise ValueError("k must be odd")
    p = np.asarray(p, dtype=float)
    cdf = np.cumsum(p)
    need = (k + 1) // 2
    med_cdf = binom.sf(need - 1, k, np.clip(cdf, 0.0, 1.0))
    med_cdf[-1] = 1.0
    out = np.diff(np.concatenate([[0.0], med_cdf]))
    return np.clip(out, 0.0, None)


def _pmf(ratings: List[int]) -> np.ndarray:
    return np.bincount(np.asarray(ratings, dtype=int), minlength=6)[1:6] / float(len(ratings))


def _vote_stats(r0: List[int], r1: List[int], r2: List[int], k: int) -> Dict[str, float]:
    p0, p1, p2 = (median_pmf(_pmf(r), k) for r in (r0, r1, r2))
    n = 1.0 - float(np.sum(p0 ** 2))
    f1 = 1.0 - float(np.sum(p1 * p0))
    f2 = 1.0 - float(np.sum(p2 * p0))
    return {"N": n, "F1": f1, "F2": f2, "E1": f1 - n, "E2": f2 - n, "Sel": (f2 - n) - (f1 - n)}


_VOTE_KEYS = ("N", "F1", "F2", "E1", "E2", "Sel")


def vote(main_path: str, ks: Iterable[int] = (1, 3, 5), n_boot: int = 2000, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Plug-in metrics on the median-of-k verdict, per cell and pooled per model, with run-level bootstrap."""
    df = A.load_runs(main_path)
    rows, samples = [], {}
    for (model, ticker), g in df.groupby(["model", "ticker"], sort=True):
        r = {c: g[g.condition == c]["rating"].astype(int).tolist() for c in ("C0", "C1", "C2")}
        if not (r["C0"] and r["C1"] and r["C2"]):
            continue
        rng = np.random.default_rng(seed)
        boots = [{c: [r[c][i] for i in rng.integers(0, len(r[c]), len(r[c]))] for c in r} for _ in range(n_boot)]
        for k in ks:
            point = _vote_stats(r["C0"], r["C1"], r["C2"], k)
            arr = {key: np.full(n_boot, np.nan) for key in _VOTE_KEYS}
            for b, bb in enumerate(boots):
                st = _vote_stats(bb["C0"], bb["C1"], bb["C2"], k)
                for key in _VOTE_KEYS:
                    arr[key][b] = st[key]
            samples[(k, model, ticker)] = arr
            row = {"k": int(k), "model": model, "ticker": ticker, "n0": len(r["C0"]), "n1": len(r["C1"]), "n2": len(r["C2"])}
            for key in _VOTE_KEYS:
                row[key], row[key + "_lo"], row[key + "_hi"] = M.summarize_samples(point[key], arr[key])
            rows.append(row)
    cells = pd.DataFrame(rows)
    prow = []
    for (k, model), g in cells.groupby(["k", "model"], sort=True):
        row = {"k": int(k), "model": model, "n_tickers": len(g)}
        for key in _VOTE_KEYS:
            stacked = np.nanmean(np.vstack([samples[(k, model, t)][key] for t in g["ticker"]]), axis=0)
            row[key], row[key + "_lo"], row[key + "_hi"] = M.summarize_samples(float(g[key].mean()), stacked)
        prow.append(row)
    pooled = pd.DataFrame(prow).sort_values(["model", "k"]).reset_index(drop=True)
    pooled = pooled.sort_values(["k", "model"]).reset_index(drop=True)
    return cells, pooled


# ---------------------------------------------------------------- Extension F: sceptical prompt

def skeptical_comparison(main_pooled: pd.DataFrame, skep_pooled: pd.DataFrame) -> pd.DataFrame:
    """Per model: point values under both prompts, their differences, and the percentile interval of the
    paired resample difference Sel'[b] - Sel[b] (pooled samples averaged over tickers, resample by resample)."""
    def pooled_samples(pooled, model, key):
        s = pooled.attrs["samples"]
        arrs = [v[key] for (m, _t), v in s.items() if m == model and key in v]
        return np.nanmean(np.vstack(arrs), axis=0) if arrs else None

    rows = []
    for model in _model_order(set(main_pooled["model"]) & set(skep_pooled["model"])):
        a = main_pooled[main_pooled.model == model].iloc[0]
        b = skep_pooled[skep_pooled.model == model].iloc[0]
        row = {"model": model}
        for key in ("N", "E1", "E2", "S1", "S2", "A1", "A2", "Sel"):
            row[key + "_main"], row[key + "_skep"] = float(a[key]), float(b[key])
            row["d" + key] = float(b[key]) - float(a[key])
        sa, sb = pooled_samples(main_pooled, model, "Sel"), pooled_samples(skep_pooled, model, "Sel")
        if sa is not None and sb is not None:
            n = min(len(sa), len(sb))
            _, row["dSel_lo"], row["dSel_hi"] = M.summarize_samples(row["dSel"], sb[:n] - sa[:n])
        else:
            row["dSel_lo"] = row["dSel_hi"] = float("nan")
        row["dSel_interval_has_zero"] = bool(row["dSel_lo"] <= 0.0 <= row["dSel_hi"])
        rows.append(row)
    return pd.DataFrame(rows)


def marker_check(paths: Dict[str, str]) -> pd.DataFrame:
    """Share of valid runs whose rationale or driver claims mention 'synthetic', per setting, model, condition."""
    rows = []
    for setting, path in paths.items():
        df = A.load_runs(path)
        for (model, cond), g in df.groupby(["model", "condition"], sort=True):
            rows.append({"setting": setting, "model": model, "condition": cond, "n": len(g),
                         "share": float(g["mentions_synthetic"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- outputs

def write_settings_table(pooled_settings: pd.DataFrame, path: str) -> None:
    cols = [("N", "$N$"), ("E1", "$E_1$"), ("E2", "$E_2$"), ("E3", "$E_3$"), ("Sel", "$\\Sigma$")]
    lines = ["\\begin{tabular}{@{}ll" + "r" * len(cols) + "l@{}}", "\\toprule",
             "Model & Setting & " + " & ".join(h for _, h in cols) + " & Diagnosis \\\\", "\\midrule"]
    for model in _model_order(pooled_settings["model"]):
        first = True
        for setting in SETTINGS:
            r = pooled_settings[(pooled_settings.model == model) & (pooled_settings.setting == setting)]
            if r.empty:
                continue
            r = r.iloc[0]
            vals = []
            for k, _ in cols:
                x = r[k]
                vals.append("--" if (x is None or (isinstance(x, float) and math.isnan(x))) else A._fmt(x, r[k + "_lo"], r[k + "_hi"]))
            lines.append("%s & %s & %s & %s \\\\" % (A.MODEL_LABEL.get(model, model) if first else "", SETTING_LABEL[setting],
                                                     " & ".join(vals), r["diagnosis"]))
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_vote_table(vote_pooled: pd.DataFrame, path: str) -> None:
    cols = [("N", "$N^{(k)}$"), ("E1", "$E_1^{(k)}$"), ("E2", "$E_2^{(k)}$"), ("Sel", "$\\Sigma^{(k)}$")]
    lines = ["\\begin{tabular}{@{}lr" + "r" * len(cols) + "@{}}", "\\toprule",
             "Model & $k$ & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
    for model in _model_order(vote_pooled["model"]):
        first = True
        for _, r in vote_pooled[vote_pooled.model == model].sort_values("k").iterrows():
            vals = [A._fmt(r[k], r[k + "_lo"], r[k + "_hi"]) for k, _ in cols]
            lines.append("%s & %d & %s \\\\" % (A.MODEL_LABEL.get(model, model) if first else "", int(r["k"]), " & ".join(vals)))
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_marker_table(marker: pd.DataFrame, path: str) -> None:
    conds = ["C0", "C1", "C2"]
    settings = [s for s in ("medium", "skeptical") if s in set(marker["setting"])]
    head = " & ".join("%s %s" % (SETTING_LABEL[s], c) for s in settings for c in conds)
    lines = ["\\begin{tabular}{@{}l" + "r" * (len(settings) * len(conds)) + "@{}}", "\\toprule",
             "Model & " + head + " \\\\", "\\midrule"]
    for model in _model_order(marker["model"]):
        vals = []
        for s in settings:
            for c in conds:
                r = marker[(marker.setting == s) & (marker.model == model) & (marker.condition == c)]
                vals.append(A._num(r.iloc[0]["share"]) if not r.empty else "--")
        lines.append("%s & %s \\\\" % (A.MODEL_LABEL.get(model, model), " & ".join(vals)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_ext2_numbers(pooled_settings: pd.DataFrame, vote_pooled: pd.DataFrame, vote_cells: pd.DataFrame,
                       comp: pd.DataFrame, marker: pd.DataFrame, path: str, eps: float = 0.05) -> None:
    lines = ["% Generated by src/analyze_ext2.py; do not edit by hand."]

    def put(name, value):
        lines.append("\\newcommand{\\%s}{%s}" % (name, value))

    for _, r in pooled_settings.iterrows():
        m, sw = A._macro_name(r["model"]), SETTING_WORD[r["setting"]]
        if r["setting"] == "medium":
            put("m%sSel" % m, A._num(r["Sel"])), put("m%sSelLo" % m, A._num(r["Sel_lo"])), put("m%sSelHi" % m, A._num(r["Sel_hi"]))
        if r["setting"] in ("medium", "high"):
            put("x%s%sSel" % (m, sw), A._num(r["Sel"])), put("x%s%sSelLo" % (m, sw), A._num(r["Sel_lo"])), put("x%s%sSelHi" % (m, sw), A._num(r["Sel_hi"]))
        if r["setting"] == "skeptical":
            for key, word in (("N", "Noise"), ("E1", "Eone"), ("E2", "Etwo"), ("S1", "Sone"), ("S2", "Stwo"), ("A1", "Aone"), ("A2", "Atwo"), ("Sel", "Sel")):
                put("y%sSkeptical%s" % (m, word), A._num(r[key]))
                put("y%sSkeptical%sLo" % (m, word), A._num(r[key + "_lo"]))
                put("y%sSkeptical%sHi" % (m, word), A._num(r[key + "_hi"]))
            put("y%sSkepticalDiag" % m, r["diagnosis"])
            put("y%sSkepticalRuns" % m, "%d" % int(sum(r["n" + c[1]] for c in ("C0", "C1", "C2"))))
            put("y%sSkepticalConfCzero" % m, A._num(r["conf0"])), put("y%sSkepticalConfCone" % m, A._num(r["conf1"]))
    for _, r in vote_pooled.iterrows():
        m, kw = A._macro_name(r["model"]), K_WORD[int(r["k"])]
        for key, word in (("N", "Noise"), ("E1", "Eone"), ("E2", "Etwo"), ("Sel", "Sel")):
            put("y%sVote%s%s" % (m, kw, word), A._num(r[key]))
            put("y%sVote%s%sLo" % (m, kw, word), A._num(r[key + "_lo"]))
            put("y%sVote%s%sHi" % (m, kw, word), A._num(r[key + "_hi"]))
    # Pre-registered cell counts for Extension E.
    k1 = vote_cells[vote_cells.k == 1].set_index(["model", "ticker"])
    kmax = vote_cells[vote_cells.k == vote_cells.k.max()].set_index(["model", "ticker"])
    moving = [idx for idx in k1.index if k1.loc[idx, "E1"] > eps]
    rises = sum(1 for idx in moving if kmax.loc[idx, "E1"] >= k1.loc[idx, "E1"] - 1e-9)
    toward_one = sum(1 for idx in moving if kmax.loc[idx, "E1"] >= 0.9)
    noisy = [idx for idx in k1.index if k1.loc[idx, "N"] > 1e-12]
    falls = sum(1 for idx in noisy if kmax.loc[idx, "N"] < k1.loc[idx, "N"] - 1e-12)
    put("yVoteCellsMoving", "%d" % len(moving)), put("yVoteCellsNotFalling", "%d" % rises), put("yVoteCellsNearOne", "%d" % toward_one)
    put("yVoteCellsNoisy", "%d" % len(noisy)), put("yVoteCellsNoiseFalls", "%d" % falls)
    put("yVoteCells", "%d" % len(k1))
    for _, r in comp.iterrows():
        m = A._macro_name(r["model"])
        for key, word in (("E1", "Eone"), ("E2", "Etwo"), ("Sel", "Sel"), ("N", "Noise")):
            put("y%sDelta%s" % (m, word), A._num(r["d" + key]))
        put("y%sDeltaSelLo" % m, A._num(r["dSel_lo"])), put("y%sDeltaSelHi" % m, A._num(r["dSel_hi"]))
    for _, r in marker.iterrows():
        m = A._macro_name(r["model"])
        put("y%sMarker%s%s" % (m, SETTING_WORD[r["setting"]], COND_WORD.get(r["condition"], r["condition"])), A._num(r["share"]))
    if not marker.empty:
        for s in ("medium", "skeptical"):
            sub = marker[(marker.setting == s) & (marker.condition.isin(["C1", "C2"]))]
            if not sub.empty:
                put("yMarker%sAll" % SETTING_WORD[s], A._num(float((sub["share"] * sub["n"]).sum() / sub["n"].sum())))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def fig_plane(pooled_settings: pd.DataFrame, cells_settings: pd.DataFrame, path: str, eps: float = 0.05) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"claude-opus-5-5": A.PALETTE["C0"], "claude-sonnet-5-5": A.PALETTE["C2"], "claude-haiku-5-5": A.PALETTE["C1"]}
    markers = {"medium": "o", "high": "s", "skeptical": "^"}
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    lim = (-0.3, 1.05)
    ax.plot(lim, lim, linestyle="--", color="#999999", linewidth=0.8)
    ax.axvline(eps, linestyle=":", color="#999999", linewidth=0.8)
    ax.axhline(eps, linestyle=":", color="#999999", linewidth=0.8)
    ax.text(-0.27, 0.98, "resistant-\nresponsive", fontsize=6.5, color="#555555", va="top")
    ax.text(-0.27, -0.27, "insensitive", fontsize=6.5, color="#555555", va="bottom")
    ax.text(1.03, 0.98, "both move", fontsize=6.5, color="#555555", va="top", ha="right")
    ax.text(1.03, -0.27, "plant only", fontsize=6.5, color="#555555", va="bottom", ha="right")
    for r in cells_settings.itertuples(index=False):
        if r.setting != "medium":
            continue
        ax.scatter(r.E1, r.E2, s=10, color=colors.get(r.model, "#444444"), alpha=0.3, marker=markers[r.setting], linewidths=0)
    for r in pooled_settings.itertuples(index=False):
        c = colors.get(r.model, "#444444")
        ax.errorbar(r.E1, r.E2, xerr=[[max(r.E1 - r.E1_lo, 0)], [max(r.E1_hi - r.E1, 0)]],
                    yerr=[[max(r.E2 - r.E2_lo, 0)], [max(r.E2_hi - r.E2, 0)]], fmt=markers[r.setting], color=c,
                    markersize=6, capsize=2, linewidth=0.8, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlim(*lim), ax.set_ylim(*lim)
    ax.set_xlabel("$E_1$: excess change under the false plant")
    ax.set_ylabel("$E_2$: excess change under the true update")
    ax.set_aspect("equal")
    ax.grid(color="#eeeeee", linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=colors[m], marker="o", linestyle="", markersize=5, label=A.MODEL_LABEL[m]) for m in colors if m in set(pooled_settings["model"])]
    handles += [Line2D([], [], color="#444444", marker=markers[s], linestyle="", markersize=5, label=SETTING_LABEL[s]) for s in SETTINGS if s in set(pooled_settings["setting"])]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(P.ROOT, "runs", "runs.jsonl"))
    ap.add_argument("--high", default=os.path.join(P.ROOT, "runs", "ext_effort_high.jsonl"))
    ap.add_argument("--skeptical", default=os.path.join(P.ROOT, "runs", "ext_skeptical.jsonl"))
    ap.add_argument("--out", default=os.path.join(P.ROOT, "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    frames = {"medium": args.runs, "high": args.high, "skeptical": args.skeptical}
    pairs, pooled_by = [], {}
    for setting, path in frames.items():
        if not os.path.exists(path):
            continue
        cells, pooled = setting_frame(path, setting, n_boot=args.n_boot)
        pairs.append((cells, pooled))
        pooled_by[setting] = pooled
    cells_all, pooled_all = combine_settings(pairs)
    cells_all.drop(columns=[c for c in cells_all.columns if c.startswith("_")], errors="ignore").to_csv(os.path.join(args.out, "ext2_cells.csv"), index=False)
    pooled_all.to_csv(os.path.join(args.out, "ext2_settings.csv"), index=False)
    write_settings_table(pooled_all, os.path.join(args.out, "ext2_settings_table.tex"))
    fig_plane(pooled_all, cells_all, os.path.join(args.out, "fig5_plane.pdf"))
    vcells, vpooled = vote(args.runs, n_boot=args.n_boot)
    vcells.to_csv(os.path.join(args.out, "ext2_vote_cells.csv"), index=False)
    vpooled.to_csv(os.path.join(args.out, "ext2_vote.csv"), index=False)
    write_vote_table(vpooled, os.path.join(args.out, "ext2_vote_table.tex"))
    comp, marker = pd.DataFrame(), pd.DataFrame()
    if "skeptical" in pooled_by:
        comp = skeptical_comparison(pooled_by["medium"], pooled_by["skeptical"])
        comp.to_csv(os.path.join(args.out, "ext2_skeptical.csv"), index=False)
        marker = marker_check({"medium": args.runs, "skeptical": args.skeptical})
        marker.to_csv(os.path.join(args.out, "ext2_marker.csv"), index=False)
        write_marker_table(marker, os.path.join(args.out, "ext2_marker_table.tex"))
    write_ext2_numbers(pooled_all, vpooled, vcells, comp, marker, os.path.join(args.out, "numbers_ext2.tex"))
    pd.set_option("display.width", 200)
    print(pooled_all[["setting", "model", "N", "E1", "E2", "E3", "Sel", "Sel_lo", "Sel_hi", "diagnosis"]].round(3).to_string(index=False))
    print(vpooled[["k", "model", "N", "E1", "E2", "Sel"]].round(3).to_string(index=False))
    if not comp.empty:
        print(comp[["model", "E1_main", "E1_skep", "E2_main", "E2_skep", "Sel_main", "Sel_skep", "dSel", "dSel_lo", "dSel_hi"]].round(3).to_string(index=False))
        print(marker.pivot_table(index=["setting", "model"], columns="condition", values="share").round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
