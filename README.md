# Noise-Calibrated Conclusion Robustness

Code, data and manuscript for *Noise-Calibrated Conclusion Robustness: Separating
Manipulation from Variance in Financial Research Agents* (Agent Evaluation Science Fall 2026,
Diagnose track). The experiment measures how a single-call research agent's valuation verdict
moves when one evidence item is added, against a noise floor from repeated clean runs and a
matched true-update control.

**Synthetic-content notice.** `packets/*.json` and the paper's Appendix C contain synthetic
counterfactual items (`"synthetic": true`, roles `false_plant` and `noise`). They were written
for this evaluation and are not factual. The flag lives in the data and the paper only: the packet text
rendered to the model carries no marker (`src/packets.py`, `MARKER`). Nothing in this repository is investment advice.

## Layout

| Path | What it holds |
|---|---|
| `METRICS.md` | Pre-registered metric definitions, committed before the pilot and unchanged since |
| `METRICS-EXT.md` | Pre-registered extension metrics (effort replication, provenance gradient, repeats needed), committed before the extension runs |
| `METRICS-EXT2.md` | Pre-registered round-two metrics (selectivity, repeat-and-vote, sceptical prompt), committed before the sceptical-prompt runs |
| `METRICS-EXT3.md` | Pre-registered round three: the synthetic marker hidden from the model, every batch rerun, Extension G (marker visible vs hidden), committed before those runs |
| `src/analyze_ext.py` | Extension analyses: `python3 -m src.analyze_ext` reads `runs/ext_*.jsonl` and writes `results/ext_*` and `results/numbers_ext.tex` |
| `src/analyze_ext2.py` | Round-two analyses: `python3 -m src.analyze_ext2` writes `results/ext2_*`, `results/fig5_plane.pdf` and `results/numbers_ext2.tex` |
| `src/analyze_ext3.py` | Extension G: `python3 -m src.analyze_ext3` compares `runs/runs.jsonl` with `runs/labelled/runs.jsonl` and writes `results/ext3_*` and `results/numbers_ext3.tex` |
| `packets/<TICKER>.json` | Evidence items per ticker with role tags and sources (as of 2026-10-01) |
| `prompts/` | System prompt (`system.txt`), its sceptical variant (`system_skeptical.txt`, one added rule), question template, output JSON schema |
| `src/packets.py` | Renders a packet for a condition; deterministic, hashed |
| `src/run.py` | Resumable driver: one Claude API call per run, one JSONL line per run |
| `src/metrics.py` | Pairwise disagreement, noise floor, excess flip, shift, attribution, bootstrap |
| `src/analyze.py` | CSV tables, LaTeX table and number macros, figures |
| `runs/pilot*.jsonl`, `runs/runs.jsonl` | Raw run logs (pilot rounds, full run) with token usage and cost |
| `runs/ext_*.jsonl` | Extension run logs: effort high, provenance styles, sceptical prompt |
| `runs/labelled/` | The first round of every batch, in which the model saw a `(SYNTHETIC)` suffix on synthetic items; kept for Extension G, reproducible with `--marker-visible` |
| `runs/run_all.sh` | Launches the four marker-hidden batches in parallel (each resumable) |
| `results/` | Generated: `metrics.csv`, `summary.csv`, `table1.tex`, `tableA.tex`, `numbers.tex`, figures, `cost.txt` |
| `paper/` | AES LaTeX template, `main.tex`, `references.bib`, `build.sh`, `main.pdf` |
| `docs/` | Design spec, implementation plan, pilot notes |

## Reproduce

Python 3.9+ with `numpy`, `pandas`, `scipy` and `pip install -r requirements.txt`
(`anthropic==0.125.0`, `matplotlib`, `jsonschema`, `pytest`). Put an API key in `.env` as
`ANTHROPIC_API_KEY=...` (the file is gitignored).

```sh
python3 -m pytest -q                                   # unit tests, no API calls
python3 -m src.run --dry-run                           # print the first prompt
python3 -m src.run --models claude-sonnet-5-5 --tickers NBIS --repeats 3 --out runs/pilot.jsonl
python3 -m src.run --repeats 20 --out runs/runs.jsonl --concurrency 6   # full run, resumable
python3 -m src.analyze --runs runs/runs.jsonl --out results --cutoffs    # tables, figures, macros
python3 -m src.run --effort high --repeats 10 --out runs/ext_effort_high.jsonl --concurrency 4
for st in filing wire forum; do python3 -m src.run --conditions C1 --plant-style $st --repeats 10 --out runs/ext_provenance_$st.jsonl; done
python3 -m src.analyze_ext                             # extension tables, figures, macros
python3 -m src.run --system-variant skeptical --conditions C0 C1 C2 --repeats 10 --concurrency 6   # -> runs/ext_skeptical.jsonl
python3 -m src.run --system-variant skeptical --models claude-haiku-5-5 --conditions C0 --repeats 20 --concurrency 6
python3 -m src.analyze_ext2                            # round-two tables, plane figure, macros
python3 -m src.analyze_ext3                            # Extension G: marker visible (runs/labelled) vs hidden (runs/)
sh runs/run_all.sh                                     # all four marker-hidden batches at once; add --marker-visible to any command above to reproduce runs/labelled/
sh paper/build.sh                                      # needs tools/tectonic (see below)
```

The driver skips (model, ticker, condition, repeat) cells already present in the output file,
so an interrupted run resumes by re-running the same command. A run file holds one system-prompt
variant; the driver refuses to append runs of another variant to it. Runs that fail three times
(refusal, output-token cap, schema-invalid JSON) are logged with `parse_ok: false`, never
dropped or fabricated.

Models: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`, called with structured
JSON output and `effort = medium`; these models accept no `temperature`. Queried 2026-10-23.

## Data notes

- Price anchors are the September 30, 2026 closes from the Yahoo Finance chart API; shares
  outstanding are Yahoo's figures on 2026-10-23, so market capitalisations are approximate.
- Dates known only to the month are stored as `YYYY-MM`.
- The JSON schema sent to the API omits `minimum`/`maximum`/`minItems`/`maxItems`, which the
  structured-output API rejects; those constraints are checked locally in `src/run.py`.
- `fair_value_usd` was added to the output after pilot round 2 (see `docs/pilot-notes.md`) and
  is reported as exploratory.

## Building the paper

The AES template needs XeLaTeX. This repository uses the standalone
[Tectonic](https://github.com/tectonic-typesetting/tectonic) binary placed at `tools/tectonic`
(gitignored; v0.17.0 was used). `paper/aes.sty` differs from the upstream template in one
place: two `\PackageError` calls in its nameref patch are `\PackageWarning`, because Tectonic
ships a nameref version the patch does not match; only PDF bookmark anchor placement is
affected.

## Cost

See `results/cost.txt` for the total and per-model API spend of the main run ($12.88 for 960 runs,
marker hidden). The marker-hidden extension runs cost a further $18.59: $8.53 for the 480 high-effort
runs, $4.88 for the 360 provenance runs and $5.18 for the 400 sceptical-prompt runs. The first,
marker-visible round under `runs/labelled/` cost $31.46 ($12.83 + $8.41 + $4.88 + $5.34). Total
for both rounds: $62.93 (the labelled main-run figure includes the third pilot round; the two discarded
pilot rounds are extra).
