"""Extension analyses (METRICS-EXT.md): effort replication, provenance gradient, repeats needed."""
import argparse
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src import analyze as A
from src import metrics as M
from src import packets as P

STYLES = ["filing", "wire", "forum"]
STYLE_LABEL = {"filing": "Filing", "wire": "Wire", "forum": "Forum"}


# ---------------------------------------------------------------- Extension A: effort replication

def effort_comparison(main_path: str, high_path: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Pooled per-model metrics at both effort levels, one row per (effort, model)."""
    frames = []
    for effort, path in (("medium", main_path), ("high", high_path)):
        df = A.load_runs(path)
        cells = A.per_cell(df, n_boot=n_boot, seed=seed)
        pooled = A.pooled(cells)
        pooled.insert(0, "effort", effort)
        frames.append(pooled)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- Extension B: provenance gradient

def _ratings(df: pd.DataFrame, model: str, ticker: str, condition: str) -> List[int]:
    g = df[(df.model == model) & (df.ticker == ticker) & (df.condition == condition)]
    return g["rating"].astype(int).tolist()


def _style_cell(r0: List[int], r1: List[int], direction: int, n_boot: int, seed: int) -> Dict[str, float]:
    """E1(style) = D(R1_style, R0) - D(R0, R0) and signed S1, with run-level bootstrap of both samples."""
    def stats(a0, a1):
        n = M.pairwise_disagreement(a0, a0, same=True)
        f1 = M.pairwise_disagreement(a1, a0)
        s1 = (float(np.mean(a1)) - float(np.mean(a0))) * direction if a1 and a0 else float("nan")
        return n, f1, f1 - n, s1
    n, f1, e1, s1 = stats(r0, r1)
    rng = np.random.default_rng(seed)
    e1s, s1s = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    for b in range(n_boot):
        b0 = [r0[i] for i in rng.integers(0, len(r0), len(r0))] if r0 else []
        b1 = [r1[i] for i in rng.integers(0, len(r1), len(r1))] if r1 else []
        _, _, e1s[b], s1s[b] = stats(b0, b1)
    out = {"N": n, "F1": f1, "E1": e1, "S1": s1, "n0": len(r0), "n1": len(r1)}
    out["E1"], out["E1_lo"], out["E1_hi"] = M.summarize_samples(e1, e1s)
    out["S1"], out["S1_lo"], out["S1_hi"] = M.summarize_samples(s1, s1s)
    out["_e1_samples"], out["_s1_samples"] = e1s, s1s
    return out


def provenance(main_path: str, style_paths: Dict[str, str], n_boot: int = 2000, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell and pooled E1/S1 by plant style, with the main run's clean cells as reference."""
    main = A.load_runs(main_path)
    rows, samples = [], {}
    for style, path in style_paths.items():
        sdf = A.load_runs(path)
        for model in sorted(main["model"].unique()):
            for ticker in sorted(main["ticker"].unique()):
                r0 = _ratings(main, model, ticker, "C0")
                r1 = _ratings(sdf, model, ticker, "C1")
                if not r1:
                    continue
                direction = P.DIRECTION.get(ticker, 1)
                cell = _style_cell(r0, r1, direction, n_boot, seed)
                samples[(style, model, ticker)] = (cell.pop("_e1_samples"), cell.pop("_s1_samples"))
                rows.append(dict(style=style, model=model, ticker=ticker, **cell))
    cells = pd.DataFrame(rows)
    prow = []
    for (style, model), g in cells.groupby(["style", "model"], sort=True):
        e_arr = np.nanmean(np.vstack([samples[(style, model, t)][0] for t in g["ticker"]]), axis=0)
        s_arr = np.nanmean(np.vstack([samples[(style, model, t)][1] for t in g["ticker"]]), axis=0)
        e1, e_lo, e_hi = M.summarize_samples(float(g["E1"].mean()), e_arr)
        s1, s_lo, s_hi = M.summarize_samples(float(g["S1"].mean()), s_arr)
        prow.append(dict(style=style, model=model, n_tickers=len(g), E1=e1, E1_lo=e_lo, E1_hi=e_hi,
                         S1=s1, S1_lo=s_lo, S1_hi=s_hi, n1=int(g["n1"].sum())))
    return cells, pd.DataFrame(prow)


# ---------------------------------------------------------------- Extension C: repeats needed

def repeats_needed(main_path: str, ns: Iterable[int] = (1, 3, 5, 10, 20), n_draw: int = 2000, seed: int = 0) -> pd.DataFrame:
    """For each model and n: draw n runs per condition from every cell, compute E1 (n >= 2) or the raw
    one-run flip (n = 1); report the mean half-width of the 95% interval of the cell-mean E1 and the share of
    draws whose pooled E1 has the same sign as the full-data pooled E1."""
    df = A.load_runs(main_path)
    rng = np.random.default_rng(seed)
    out = []
    for model in sorted(df["model"].unique()):
        tickers = sorted(df[df.model == model]["ticker"].unique())
        full = []
        cell_r = {}
        for t in tickers:
            r0, r1 = _ratings(df, model, t, "C0"), _ratings(df, model, t, "C1")
            cell_r[t] = (r0, r1)
            if len(r0) >= 2 and r1:
                full.append(M.pairwise_disagreement(r1, r0) - M.pairwise_disagreement(r0, r0, same=True))
        full_sign = np.sign(np.nanmean(full)) if full else 0.0
        for n in ns:
            draws = np.full(n_draw, np.nan)
            for b in range(n_draw):
                vals = []
                for t in tickers:
                    r0, r1 = cell_r[t]
                    if not r0 or not r1:
                        continue
                    d0 = [r0[i] for i in rng.integers(0, len(r0), n)]
                    d1 = [r1[i] for i in rng.integers(0, len(r1), n)]
                    if n == 1:
                        vals.append(1.0 if d1[0] != d0[0] else 0.0)  # what a one-run study records
                    else:
                        vals.append(M.pairwise_disagreement(d1, d0) - M.pairwise_disagreement(d0, d0, same=True))
                draws[b] = np.mean(vals) if vals else np.nan
            good = draws[~np.isnan(draws)]
            lo, hi = (np.percentile(good, 2.5), np.percentile(good, 97.5)) if len(good) else (np.nan, np.nan)
            agree = float(np.mean(np.sign(good) == full_sign)) if len(good) and full_sign != 0 else float("nan")
            out.append(dict(model=model, n=int(n), E1_mean=float(np.mean(good)) if len(good) else np.nan,
                            E1_halfwidth=float((hi - lo) / 2.0), E1_sign_agreement=agree,
                            statistic="raw flip (no noise floor)" if n == 1 else "E1"))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- outputs

_WORD = {"1": "one", "2": "two", "3": "three", "5": "five", "10": "ten", "20": "twenty"}
_NUMWORD = {1: "One", 3: "Three", 5: "Five", 10: "Ten", 20: "Twenty"}


def write_ext_numbers(effort_df: pd.DataFrame, prov_pooled: pd.DataFrame, rep: pd.DataFrame, path: str) -> None:
    lines = ["% Generated by src/analyze_ext.py; do not edit by hand."]
    for _, r in effort_df.iterrows():
        m = A._macro_name(r["model"])
        lvl = "High" if r["effort"] == "high" else "Med"
        for k in A.METRIC_COLS:
            word = {"N": "Noise", "F": "F", "E": "E", "S": "S", "A": "A"}[k[0]]
            idx = _WORD.get(k[1:], "")
            lines.append("\\newcommand{\\x%s%s%s%s}{%s}" % (m, lvl, word, idx, A._num(r[k])))
            lines.append("\\newcommand{\\x%s%s%s%sLo}{%s}" % (m, lvl, word, idx, A._num(r[k + "_lo"])))
            lines.append("\\newcommand{\\x%s%s%s%sHi}{%s}" % (m, lvl, word, idx, A._num(r[k + "_hi"])))
        lines.append("\\newcommand{\\x%s%sDiag}{%s}" % (m, lvl, r["diagnosis"]))
        lines.append("\\newcommand{\\x%s%sRuns}{%d}" % (m, lvl, int(sum(r["n" + c[1]] for c in P.CONDITIONS))))
    for _, r in prov_pooled.iterrows():
        m = A._macro_name(r["model"])
        st = STYLE_LABEL[r["style"]]
        for k in ("E1", "S1"):
            idx = _WORD[k[1]]
            lines.append("\\newcommand{\\x%s%s%s%s}{%s}" % (m, st, k[0], idx, A._num(r[k])))
            lines.append("\\newcommand{\\x%s%s%s%sLo}{%s}" % (m, st, k[0], idx, A._num(r[k + "_lo"])))
            lines.append("\\newcommand{\\x%s%s%s%sHi}{%s}" % (m, st, k[0], idx, A._num(r[k + "_hi"])))
    for _, r in rep.iterrows():
        m = A._macro_name(r["model"])
        nw = _NUMWORD[int(r["n"])]
        lines.append("\\newcommand{\\x%sRep%sHalfwidth}{%s}" % (m, nw, A._num(r["E1_halfwidth"])))
        lines.append("\\newcommand{\\x%sRep%sAgree}{%s}" % (m, nw, A._num(r["E1_sign_agreement"])))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_effort_table(effort_df: pd.DataFrame, path: str) -> None:
    cols = [("N", "$N$"), ("E1", "$E_1$"), ("E2", "$E_2$"), ("E3", "$E_3$")]
    lines = ["\\begin{tabular}{@{}ll" + "r" * len(cols) + "l@{}}", "\\toprule",
             "Model & Effort & " + " & ".join(h for _, h in cols) + " & Diagnosis \\\\", "\\midrule"]
    for model in sorted(effort_df["model"].unique(), key=lambda m: list(A.MODEL_LABEL).index(m) if m in A.MODEL_LABEL else 99):
        for effort in ("medium", "high"):
            r = effort_df[(effort_df.model == model) & (effort_df.effort == effort)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = [A._fmt(r[k], r[k + "_lo"], r[k + "_hi"]) for k, _ in cols]
            lines.append("%s & %s & %s & %s \\\\" % (A.MODEL_LABEL.get(model, model), effort, " & ".join(cells), r["diagnosis"]))
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_provenance_table(prov_pooled: pd.DataFrame, main_pooled: pd.DataFrame, path: str) -> None:
    """Rows = models; columns = E1 for the original plant and for each style."""
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "Model & Original plant & Filing & Wire & Forum \\\\", "\\midrule"]
    for model in sorted(prov_pooled["model"].unique(), key=lambda m: list(A.MODEL_LABEL).index(m) if m in A.MODEL_LABEL else 99):
        mp = main_pooled[main_pooled.model == model].iloc[0]
        vals = [A._fmt(mp["E1"], mp["E1_lo"], mp["E1_hi"])]
        for st in STYLES:
            r = prov_pooled[(prov_pooled["model"] == model) & (prov_pooled["style"] == st)]
            vals.append(A._fmt(r.iloc[0]["E1"], r.iloc[0]["E1_lo"], r.iloc[0]["E1_hi"]) if not r.empty else "--")
        lines.append("%s & %s \\\\" % (A.MODEL_LABEL.get(model, model), " & ".join(vals)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def fig_effort(effort_df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    models = [m for m in A.MODEL_LABEL if m in set(effort_df["model"])]
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.7), sharey=True)
    for ax, (k, title) in zip(axes, (("N", "noise floor $N$"), ("E1", "$E_1$ false plant"), ("E2", "$E_2$ true update"), ("E3", "$E_3$ immaterial"))):
        x = np.arange(len(models))
        for j, (effort, marker, fill) in enumerate((("medium", "o", "full"), ("high", "s", "none"))):
            sub = effort_df[effort_df.effort == effort].set_index("model").reindex(models)
            y, lo, hi = sub[k].values, sub[k + "_lo"].values, sub[k + "_hi"].values
            ax.errorbar(x + (j - 0.5) * 0.25, y, yerr=np.vstack([np.nan_to_num(y - lo), np.nan_to_num(hi - y)]), fmt=marker,
                        color=A.PALETTE["C0"] if effort == "medium" else A.PALETTE["C1"], fillstyle=fill, markersize=5, capsize=2, linewidth=1,
                        label="effort " + effort)
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([A.MODEL_LABEL[m].split()[0] for m in models])
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("pooled value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_repeats(rep: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.0, 1.7))
    colors = {"claude-opus-5-5": A.PALETTE["C0"], "claude-sonnet-5-5": A.PALETTE["C2"], "claude-haiku-5-5": A.PALETTE["C1"]}
    for model, g in rep.groupby("model"):
        g = g[g.n >= 2].sort_values("n")
        ax.plot(g["n"], g["E1_halfwidth"], marker="o", markersize=4, linewidth=1.2, color=colors.get(model, "#444444"),
                label=A.MODEL_LABEL.get(model, model))
    ax.set_xlabel("repeats per condition")
    ax.set_ylabel("95% half-width on $E_1$")
    ax.set_xticks([3, 5, 10, 20])
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(P.ROOT, "runs", "runs.jsonl"))
    ap.add_argument("--high", default=os.path.join(P.ROOT, "runs", "ext_effort_high.jsonl"))
    ap.add_argument("--provenance-dir", default=os.path.join(P.ROOT, "runs"))
    ap.add_argument("--out", default=os.path.join(P.ROOT, "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    eff = effort_comparison(args.runs, args.high, n_boot=args.n_boot)
    eff.to_csv(os.path.join(args.out, "ext_effort.csv"), index=False)
    write_effort_table(eff, os.path.join(args.out, "ext_effort_table.tex"))
    fig_effort(eff, os.path.join(args.out, "fig3_effort.pdf"))
    style_paths = {s: os.path.join(args.provenance_dir, "ext_provenance_%s.jsonl" % s) for s in STYLES}
    style_paths = {s: p for s, p in style_paths.items() if os.path.exists(p)}
    cells, pooled = provenance(args.runs, style_paths, n_boot=args.n_boot)
    cells.to_csv(os.path.join(args.out, "ext_provenance_cells.csv"), index=False)
    pooled.to_csv(os.path.join(args.out, "ext_provenance.csv"), index=False)
    main_pooled = eff[eff.effort == "medium"]
    write_provenance_table(pooled, main_pooled, os.path.join(args.out, "ext_provenance_table.tex"))
    rep = repeats_needed(args.runs)
    rep.to_csv(os.path.join(args.out, "ext_repeats.csv"), index=False)
    fig_repeats(rep, os.path.join(args.out, "fig4_repeats.pdf"))
    write_ext_numbers(eff, pooled, rep, os.path.join(args.out, "numbers_ext.tex"))
    pd.set_option("display.width", 200)
    print(eff[["effort", "model", "N", "E1", "E2", "E3", "diagnosis"]].round(3).to_string(index=False))
    print(pooled[["style", "model", "E1", "E1_lo", "E1_hi", "S1"]].round(3).to_string(index=False))
    print(rep.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
