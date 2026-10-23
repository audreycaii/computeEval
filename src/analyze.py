"""Turn runs/*.jsonl into per-cell and pooled metrics, a LaTeX table and figures."""
import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src import metrics as M
from src import packets as P

PALETTE = {"C0": "#0072b2", "C1": "#d55e00", "C2": "#009e73", "C3": "#e69f00"}  # validated, CVD-safe
COND_LABEL = {"C0": "C0 clean", "C1": "C1 false plant", "C2": "C2 true update", "C3": "C3 noise"}
MODEL_LABEL = {"claude-opus-5-5": "Opus 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "claude-haiku-5-5": "Haiku 5.5"}
METRIC_COLS = ["N", "F1", "F2", "F3", "E1", "E2", "E3", "S1", "S2", "S3", "A1", "A2", "A3"]
FV_COLS = ["V1", "V2", "V3"]  # exploratory: fair-value shift, % of C0 mean, signed by direction for C1/C2


def load_runs(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("parsed") or {}
            cited = [e for d in p.get("drivers", []) for e in d.get("evidence_ids", [])]
            rows.append({
                "run_id": r["run_id"], "model": r["model"], "ticker": r["ticker"], "condition": r["condition"],
                "direction": r["direction"], "repeat": r["repeat"], "special_item_id": r["special_item_id"],
                "parse_ok": bool(r["parse_ok"]), "rating": p.get("rating"), "confidence": p.get("confidence"),
                "fair_value_usd": p.get("fair_value_usd"),
                "attributed": bool(r["special_item_id"] is not None and r["special_item_id"] in cited),
                "hallucinated_ids": len(r.get("unknown_ids") or []), "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"], "cost_usd": r["cost_usd"], "error": r.get("error"),
                "attempts": r.get("attempts"),
            })
    df = pd.DataFrame(rows)
    df["attributed"] = df["attributed"].astype(bool)
    return df[df["parse_ok"]].reset_index(drop=True)


def load_all_rows(path: str) -> pd.DataFrame:
    """All records including failed ones (for counts and cost)."""
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    return pd.DataFrame([{k: r.get(k) for k in ("model", "ticker", "condition", "parse_ok", "cost_usd", "error", "attempts", "input_tokens", "output_tokens")} for r in recs])


def _cell_inputs(df: pd.DataFrame):
    ratings = {c: df[df.condition == c]["rating"].astype(int).tolist() for c in P.CONDITIONS}
    attributed = {c: df[df.condition == c]["attributed"].tolist() for c in P.CONDITIONS}
    fv = {c: df[df.condition == c]["fair_value_usd"].astype(float).tolist() for c in P.CONDITIONS}
    return ratings, attributed, fv


def _fv_shift(fv: Dict[str, List[float]], direction: int) -> Dict[str, float]:
    base = np.mean(fv["C0"]) if fv["C0"] else float("nan")
    out = {}
    for k in ("C1", "C2", "C3"):
        i = k[1]
        if not fv[k] or math.isnan(base) or base == 0:
            out["V" + i] = float("nan")
            continue
        pct = (np.mean(fv[k]) - base) / base * 100.0
        out["V" + i] = pct * direction if k != "C3" else pct
    return out


def _fv_bootstrap(fv: Dict[str, List[float]], direction: int, n_boot: int, seed: int) -> Dict[str, tuple]:
    rng = np.random.default_rng(seed + 1)
    point = _fv_shift(fv, direction)
    samples = {k: [] for k in point}
    for _ in range(n_boot):
        rs = {c: ([v[i] for i in rng.integers(0, len(v), size=len(v))] if v else []) for c, v in fv.items()}
        s = _fv_shift(rs, direction)
        for k in point:
            samples[k].append(s[k])
    out = {}
    for k in point:
        arr = np.array([v for v in samples[k] if not math.isnan(v)])
        out[k] = (point[k], float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) and not math.isnan(point[k]) else (point[k], float("nan"), float("nan"))
    return out


def per_cell(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for (model, ticker), g in df.groupby(["model", "ticker"], sort=True):
        ratings, attributed, fv = _cell_inputs(g)
        direction = P.DIRECTION.get(ticker, int(g["direction"].iloc[0]))
        boot = M.bootstrap_cell(ratings, attributed, direction, n_boot=n_boot, seed=seed)
        fvb = _fv_bootstrap(fv, direction, n_boot, seed)
        point = M.cell_metrics(ratings, attributed, direction)
        row = {"model": model, "ticker": ticker, "direction": direction}
        for k in METRIC_COLS:
            row[k], row[k + "_lo"], row[k + "_hi"] = boot[k]
        for k in FV_COLS:
            row[k], row[k + "_lo"], row[k + "_hi"] = fvb[k]
        for c in P.CONDITIONS:
            row["n" + c[1]] = point["n" + c[1]]
            row["conf" + c[1]] = float(np.mean(g[g.condition == c]["confidence"])) if (g.condition == c).any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def pooled(cells: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per-model mean over tickers; CI from the ticker-level intervals' half-widths combined as a mean
    (approximation: ticker cells are independent, so the variance of the mean is the mean variance / k)."""
    rows = []
    for model, g in cells.groupby("model", sort=True):
        row = {"model": model, "n_tickers": len(g)}
        for k in METRIC_COLS + FV_COLS:
            vals = g[k].astype(float)
            row[k] = float(vals.mean()) if vals.notna().any() else float("nan")
            half = ((g[k + "_hi"] - g[k + "_lo"]) / 2.0).astype(float)
            valid = half.notna()
            if valid.any():
                se = math.sqrt(float((half[valid] ** 2).sum())) / valid.sum()
                row[k + "_lo"], row[k + "_hi"] = row[k] - se, row[k] + se
            else:
                row[k + "_lo"] = row[k + "_hi"] = float("nan")
        for c in P.CONDITIONS:
            row["n" + c[1]] = int(g["n" + c[1]].sum())
            row["conf" + c[1]] = float(g["conf" + c[1]].mean())
        row["diagnosis"] = M.diagnosis(row["E1"], row["E2"])
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(x: float, lo: float, hi: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = ("%." + str(digits) + "f") % x
    if not (math.isnan(lo) or math.isnan(hi)):
        s += " {\\scriptsize[%s, %s]}" % (("%." + str(digits) + "f") % lo, ("%." + str(digits) + "f") % hi)
    return s


def write_table1(pooled_df: pd.DataFrame, path: str) -> None:
    cols = [("N", "$N$"), ("F1", "$F_1$"), ("E1", "$E_1$"), ("E2", "$E_2$"), ("E3", "$E_3$"),
            ("S1", "$S_1$"), ("S2", "$S_2$")]
    lines = ["\\begin{tabular}{@{}l" + "r" * len(cols) + "l@{}}", "\\toprule",
             "Model & " + " & ".join(h for _, h in cols) + " & Diagnosis \\\\", "\\midrule"]
    for _, r in pooled_df.iterrows():
        cells = [_fmt(r[k], r[k + "_lo"], r[k + "_hi"]) for k, _ in cols]
        lines.append("%s & %s & %s \\\\" % (MODEL_LABEL.get(r["model"], r["model"]), " & ".join(cells), r["diagnosis"]))
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _tex_escape(t: str) -> str:
    return t.replace("\\", "/").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")


def write_table_cells(cells: pd.DataFrame, path: str) -> None:
    """Appendix table: one row per model x ticker with the headline metrics and run counts."""
    cols = ["N", "E1", "E2", "E3", "S1", "S2", "V1", "V2", "A1", "A2"]
    heads = ["$N$", "$E_1$", "$E_2$", "$E_3$", "$S_1$", "$S_2$", "$V_1$", "$V_2$", "$A_1$", "$A_2$"]
    lines = ["\\begin{tabular}{@{}llr" + "r" * len(cols) + "@{}}", "\\toprule",
             "Model & Ticker & $n_0/n_1/n_2/n_3$ & " + " & ".join(heads) + " \\\\", "\\midrule"]
    for _, r in cells.sort_values(["model", "ticker"]).iterrows():
        ns = "%d/%d/%d/%d" % (r["n0"], r["n1"], r["n2"], r["n3"])
        vals = [_num(r[k], 1 if k.startswith("V") else 2) for k in cols]
        lines.append("%s & %s & %s & %s \\\\" % (MODEL_LABEL.get(r["model"], r["model"]), r["ticker"], ns, " & ".join(vals)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_packet_table(path: str) -> None:
    """Appendix long table listing every packet item: id, date, role, source and the first sentence."""
    lines = ["\\begin{aeslongtable}{@{}p{1.5cm}p{2.0cm}p{2.3cm}p{8.4cm}@{}}{4}",
             "  {Every evidence item in the four packets. Role: base items appear in every condition; the "
             "true update (C2), false plant (C1) and noise item (C3) appear only in their condition. Items "
             "marked synthetic are counterfactual and were written for this evaluation; they are not factual.}",
             "  {tab:packets}", "  {ID & Date & Role & Source and first sentence}"]
    for t in P.TICKERS:
        for it in sorted(P.load_packet(t)["items"], key=lambda x: x["id"]):
            role = it["role"].replace("_", " ") + (" (synthetic)" if it["synthetic"] else "")
            first = it["text"].split(". ")[0].rstrip(".") + "."
            lines.append("%s & %s & %s & \\textit{%s.} %s\\\\" % (it["id"], it["date"], role, _tex_escape(it["source"]), _tex_escape(first)))
    lines.append("\\end{aeslongtable}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def fig_ratings(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    models = sorted(df["model"].unique(), key=lambda m: list(MODEL_LABEL).index(m) if m in MODEL_LABEL else 99)
    tickers = [t for t in P.TICKERS if t in set(df["ticker"])] or sorted(df["ticker"].unique())
    fig, axes = plt.subplots(len(models), len(tickers), figsize=(6.3, 1.6 * len(models) + 0.5), squeeze=False, sharex=True, sharey=True)
    width = 0.2
    for i, m in enumerate(models):
        for j, t in enumerate(tickers):
            ax = axes[i][j]
            g = df[(df.model == m) & (df.ticker == t)]
            for k, c in enumerate(P.CONDITIONS):
                r = g[g.condition == c]["rating"]
                counts = [int((r == v).sum()) for v in range(1, 6)]
                share = [x / len(r) if len(r) else 0 for x in counts]
                ax.bar(np.arange(1, 6) + (k - 1.5) * width, share, width=width, color=PALETTE[c], label=COND_LABEL[c], linewidth=0)
            ax.set_xticks(range(1, 6))
            ax.set_ylim(0, 1)
            ax.grid(axis="y", color="#dddddd", linewidth=0.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(t, fontsize=9)
            if j == 0:
                ax.set_ylabel(MODEL_LABEL.get(m, m) + "\nshare of runs", fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("rating (1--5)", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_excess(pooled_df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    models = list(pooled_df["model"])
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    x = np.arange(len(models))
    off = {"C1": -0.22, "C2": 0.0, "C3": 0.22}
    for c in ("C1", "C2", "C3"):
        i = c[1]
        e = pooled_df["E" + i].astype(float).values
        lo = pooled_df["E" + i + "_lo"].astype(float).values
        hi = pooled_df["E" + i + "_hi"].astype(float).values
        f = pooled_df["F" + i].astype(float).values
        yerr = np.vstack([np.nan_to_num(e - lo), np.nan_to_num(hi - e)])
        ax.errorbar(x + off[c], e, yerr=yerr, fmt="o", color=PALETTE[c], markersize=5, capsize=2, linewidth=1, label="$E$ " + COND_LABEL[c])
        ax.scatter(x + off[c], f, facecolors="none", edgecolors=PALETTE[c], s=28, linewidths=1, label="$F$ (raw) " + COND_LABEL[c], zorder=3)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models])
    ax.set_ylabel("flip rate vs. clean runs")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_cost(all_rows: pd.DataFrame, path: str) -> None:
    lines = ["total_usd %.4f" % all_rows["cost_usd"].sum(), "total_runs %d" % len(all_rows),
             "valid_runs %d" % int(all_rows["parse_ok"].sum())]
    for m, g in all_rows.groupby("model"):
        lines.append("%s usd %.4f runs %d valid %d retries %d" % (m, g["cost_usd"].sum(), len(g), int(g["parse_ok"].sum()), int((g["attempts"].fillna(1) - 1).sum())))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _macro_name(model: str) -> str:
    """LaTeX macro stem for a model id: letters only, e.g. claude-opus-5-5 -> Opus."""
    short = MODEL_LABEL.get(model, model).split()[0]
    return "".join(ch for ch in short if ch.isalpha())


def _num(x, digits=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else ("%." + str(digits) + "f") % x


def write_numbers(pooled_df: pd.DataFrame, cells: pd.DataFrame, all_rows: pd.DataFrame, path: str) -> None:
    """Every number the paper quotes, as LaTeX macros generated from the CSVs."""
    lines = ["% Generated by src/analyze.py; do not edit by hand.",
             "\\newcommand{\\nTotalRuns}{%d}" % len(all_rows),
             "\\newcommand{\\nValidRuns}{%d}" % int(all_rows["parse_ok"].sum()),
             "\\newcommand{\\nFailedRuns}{%d}" % int((~all_rows["parse_ok"].astype(bool)).sum()),
             "\\newcommand{\\costTotal}{%.2f}" % all_rows["cost_usd"].sum(),
             "\\newcommand{\\nRetries}{%d}" % int((all_rows["attempts"].fillna(1) - 1).sum())]
    for _, r in pooled_df.iterrows():
        m = _macro_name(r["model"])
        for k in METRIC_COLS + FV_COLS:
            word = {"N": "Noise", "F": "F", "E": "E", "S": "S", "A": "A", "V": "V"}[k[0]]
            idx = {"1": "one", "2": "two", "3": "three"}.get(k[1:], "")
            lines.append("\\newcommand{\\m%s%s%s}{%s}" % (m, word, idx, _num(r[k])))
            lines.append("\\newcommand{\\m%s%s%sLo}{%s}" % (m, word, idx, _num(r[k + "_lo"])))
            lines.append("\\newcommand{\\m%s%s%sHi}{%s}" % (m, word, idx, _num(r[k + "_hi"])))
        lines.append("\\newcommand{\\m%sEoneNet}{%s}" % (m, _num(r["E1"] - r["E3"])))
        lines.append("\\newcommand{\\m%sEtwoNet}{%s}" % (m, _num(r["E2"] - r["E3"])))
        lines.append("\\newcommand{\\m%sDiag}{%s}" % (m, r["diagnosis"]))
        for c in P.CONDITIONS:
            word = {"0": "zero", "1": "one", "2": "two", "3": "three"}[c[1]]
            lines.append("\\newcommand{\\m%sConfC%s}{%s}" % (m, word, _num(r["conf" + c[1]])))
        lines.append("\\newcommand{\\m%sRuns}{%d}" % (m, int(sum(r["n" + c[1]] for c in P.CONDITIONS))))
    for _, r in cells.iterrows():
        m = _macro_name(r["model"])
        for k in ("N", "E1", "E2", "E3", "S1", "S2", "V1", "V2", "A1", "A2"):
            idx = {"1": "one", "2": "two", "3": "three"}.get(k[1:], "")
            lines.append("\\newcommand{\\c%s%s%s%s}{%s}" % (m, r["ticker"], k[0] if k != "N" else "Noise", idx, _num(r[k])))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def fetch_cutoffs(models: List[str], path: str) -> None:
    """Three extra API calls: each model's self-reported training cutoff (logged, not trusted)."""
    from src.run import load_env
    load_env()
    import anthropic
    client = anthropic.Anthropic()
    out = {}
    for m in models:
        resp = client.messages.create(model=m, max_tokens=4000, output_config={"effort": "low"},
                                      messages=[{"role": "user", "content": "What is your training data cutoff? Answer with a month and year only."}])
        out[m] = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(P.ROOT, "runs", "runs.jsonl"))
    ap.add_argument("--out", default=os.path.join(P.ROOT, "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cutoffs", action="store_true", help="also query each model for its stated training cutoff")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    df = load_runs(args.runs)
    cells = per_cell(df, n_boot=args.n_boot)
    pooled_df = pooled(cells, n_boot=args.n_boot)
    cells.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    pooled_df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    write_table1(pooled_df, os.path.join(args.out, "table1.tex"))
    fig_ratings(df, os.path.join(args.out, "fig1_ratings.pdf"))
    fig_excess(pooled_df, os.path.join(args.out, "fig2_excess.pdf"))
    all_rows = load_all_rows(args.runs)
    write_cost(all_rows, os.path.join(args.out, "cost.txt"))
    write_numbers(pooled_df, cells, all_rows, os.path.join(args.out, "numbers.tex"))
    write_table_cells(cells, os.path.join(args.out, "tableA.tex"))
    write_packet_table(os.path.join(args.out, "packets.tex"))
    if args.cutoffs:
        fetch_cutoffs(sorted(df["model"].unique()), os.path.join(args.out, "cutoffs.json"))
    print(pooled_df[["model", "N", "E1", "E2", "E3", "S1", "S2", "V1", "V2", "A1", "A2", "diagnosis"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
