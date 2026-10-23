# Noise-Calibrated Conclusion Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 480-call experiment measuring noise-adjusted conclusion changes of Claude research agents under planted false, true and irrelevant evidence, and produce a 4–6 page AES-format paper from the logs.

**Architecture:** Static JSON evidence packets are rendered per condition into a prompt; a resumable Python driver sends each prompt to the Claude API with structured JSON output and appends one line per run to a JSONL log; a metrics module computes pre-registered pairwise-disagreement statistics with bootstrap CIs; an analysis script turns the log into CSV tables, a LaTeX table and PDF figures that the paper includes verbatim.

**Tech Stack:** Python 3.9.6 (no `X | Y` annotations, no `match`), `anthropic` 0.125.0 (0.x line; 1.x needs 3.10), numpy/pandas/scipy already installed, `matplotlib`, `jsonschema`, pytest; Tectonic 0.17.0 (aarch64-apple-darwin) for XeLaTeX; AES LaTeX template.

**Spec:** `docs/superpowers/specs/2026-10-23-noise-calibrated-conclusion-robustness-design.md`

## Global Constraints

- Models: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`; `output_config={"format": {"type":"json_schema","schema":...}, "effort": "medium"}`; `max_tokens=8000`; no `temperature`; no tools.
- API key read from `computeEval/.env` (`ANTHROPIC_API_KEY=...`), never committed; `.env` is in `.gitignore`.
- Tickers `NBIS`, `CRWV`, `IREN`, `NVDA`; conditions `C0`, `C1`, `C2`, `C3`; directions NBIS +1, CRWV −1, IREN −1, NVDA +1.
- As-of date `2026-10-01` in every prompt. Price anchor = close on 2026-09-30 from Yahoo Finance chart API: NBIS 235.88, CRWV 87.12, IREN 40.88, NVDA 228.38 (USD); shares outstanding from Yahoo quoteSummary on 2026-10-23: NBIS 220,406,311; CRWV 458,871,690; IREN 394,058,648; NVDA 24,147,000,000; market cap in the packet = close × shares, labelled "approximate".
- Output schema: `rating` int 1–5, `confidence` number 0–1, `drivers` array of exactly 3 `{claim, evidence_ids}`, `rationale` string.
- Every synthetic item (`false_plant`, `noise`) carries `"synthetic": true`; every released file that contains one carries the notice "Contains synthetic counterfactual items for evaluation; not factual."
- `METRICS.md` is committed before the pilot and not edited after (any later change is a new commit that the paper discloses).
- Never fabricate a run; failed runs are logged with `parse_ok=false` and reported.
- Paper: `\usepackage{aes}` (public), author `Zixin Audrey Cai`, affiliation `Peira Labs`, email `zixincai@umich.edu`, 4–6 pages excluding references, deadline 2026-10-25.
- List prices for cost logging (USD per 1M tokens, input/output): opus 4/20, sonnet 2/10, haiku 0.10/0.50.

## Review Focus

1. A response with `stop_reason == "refusal"` or `max_tokens` must be logged as a failed run, not crash the driver or be parsed as empty JSON. (Task 3 test `test_refusal_logged_as_failed`.)
2. JSON that validates against the API schema but violates semantics (rating 0 or 6, drivers count ≠ 3, confidence 1.2) must be `parse_ok=false`. (Task 3 test `test_parse_rejects_out_of_range`.)
3. Killing the driver mid-run and restarting must not duplicate any (model, ticker, condition, repeat) line. (Task 3 test `test_resume_skips_completed`.)
4. A cell with fewer than 2 valid runs, or zero, must yield NaN metrics and still produce the CSV and figures. (Task 5 test `test_metrics_with_tiny_cells`, Task 7 test `test_analyze_handles_missing_cell`.)
5. An `evidence_ids` entry that is not an ID in the rendered packet must count as not-attributed and be counted in a `hallucinated_ids` column. (Task 3 test `test_parse_flags_unknown_ids`, Task 5 attribution test.)

---

### Task 1: Pre-registration, prompts and scaffold

**Files:**
- Create: `METRICS.md`, `prompts/system.txt`, `prompts/question.txt`, `prompts/schema.json`, `requirements.txt`, `src/__init__.py`, `tests/__init__.py`, `tests/test_prompts.py`

**Interfaces:**
- Produces: `prompts/schema.json` (the API-facing JSON schema; `drivers` has `minItems`/`maxItems` 3 **only** if the API accepts them in the pilot, otherwise they are enforced in `parse_output`), `prompts/question.txt` with placeholders `{company}`, `{ticker}`, `{as_of_date}`, `{packet}`.

- [ ] **Step 1: Write `METRICS.md`** containing, verbatim from spec §7: the definitions of D(A,B), N, E_k, S_k, A_k, F_k, the three-way diagnosis labels, the bootstrap procedure (2,000 resamples, percentile 95% CI, seed 0, run-level within cell, pooled = mean over tickers), and the pre-registered expectation.
- [ ] **Step 2: Write `prompts/system.txt`**: equity research analyst agent; use only the packet; every packet item is admissible regardless of date; do not use knowledge of events after the as-of date; three drivers must each cite at least one packet item ID; output only JSON matching the schema.
- [ ] **Step 3: Write `prompts/question.txt`** with the spec's question sentence and the placeholders above, then the rating scale legend (1 = strongly unattractive/overvalued … 5 = strongly attractive/undervalued).
- [ ] **Step 4: Write `prompts/schema.json`** per Global Constraints with `additionalProperties: false` at every object level and all fields `required`.
- [ ] **Step 5: Write `requirements.txt`**: `anthropic==0.125.0`, `matplotlib`, `jsonschema`, `pytest`. Install: `python3 -m pip install --user -r requirements.txt`.
- [ ] **Step 6: Write failing test `tests/test_prompts.py::test_schema_shape`**: loads schema; asserts `required == ["rating","confidence","drivers","rationale"]` (as a set), `rating` has `minimum 1`/`maximum 5`, `confidence` `minimum 0`/`maximum 1`; `test_question_placeholders` asserts all four placeholders present.
- [ ] **Step 7: Run `python3 -m pytest tests/test_prompts.py -v`** → PASS (files exist from steps 1–4; if FAIL, fix the file, not the test).
- [ ] **Step 8: Commit** `git add METRICS.md prompts requirements.txt src tests && git commit -m "Pre-register metrics; add prompts and schema"`.

### Task 2: Evidence packets

**Files:**
- Create: `packets/NBIS.json`, `packets/CRWV.json`, `packets/IREN.json`, `packets/NVDA.json`, `src/packets.py`, `tests/test_packets.py`

**Interfaces:**
- Produces: `TICKERS = ["NBIS","CRWV","IREN","NVDA"]`, `CONDITIONS = ["C0","C1","C2","C3"]`, `DIRECTION = {"NBIS":1,"CRWV":-1,"IREN":-1,"NVDA":1}`, `COMPANY = {"NBIS":"Nebius Group", ...}`; `load_packet(ticker: str) -> dict` (keys `ticker`, `as_of`, `items: list[dict]`); `special_item_id(ticker: str, condition: str) -> Optional[str]` (None for C0); `render(ticker: str, condition: str) -> Tuple[str, str]` returning `(packet_text, sha256_hex)`; `item_ids(ticker: str, condition: str) -> List[str]`; `build_prompt(ticker: str, condition: str) -> str` (question with placeholders filled).

- [ ] **Step 1: Write failing tests** in `tests/test_packets.py`:
  - `test_roles_per_ticker`: each ticker has exactly one item each of role `true_update`, `false_plant`, `noise`, and 8–12 `base`.
  - `test_special_items_only_in_their_condition`: `item_ids(t,"C1")` contains the `false_plant` id and not the other two; same for C2/C3; C0 contains none.
  - `test_render_deterministic`: `render(t,c)` called twice gives equal text and sha; shas differ across conditions.
  - `test_synthetic_flags`: `false_plant` and `noise` items have `synthetic == True` and `base`/`true_update` have `False`; every non-synthetic item has a non-empty `url`.
  - `test_price_anchor_present`: rendered C0 text contains the Sept 30 close for that ticker from Global Constraints.
  - `test_items_sorted_by_date`: ids appear in the rendered text in ascending `date` order.
- [ ] **Step 2: Run `python3 -m pytest tests/test_packets.py -v`** → FAIL (module missing).
- [ ] **Step 3: Write the four packet JSON files** from spec §5: base items (one JSON item per bullet fact, 2–4 sentences, source/url from the reading list), the `true_update`, `false_plant` and `noise` items from the table, plus a `base` item `<T>-PX` with the Sept 30 close, approximate market cap, source `Yahoo Finance historical data`, url `https://finance.yahoo.com/quote/<T>/history/`, accessed 2026-10-23. Item id format `<TICKER>-NN`. Render rule for the line: `[ID] (YYYY-MM-DD, source_type, source) text`.
- [ ] **Step 4: Implement `src/packets.py`** with the interfaces above; `render` sorts by `(date, id)`, joins lines with blank lines, prefixes a header `Evidence packet for {company} ({ticker}), as of 2026-10-01. {n} items.`; sha256 over the UTF-8 text.
- [ ] **Step 5: Run tests** → PASS.
- [ ] **Step 6: Commit** `git add packets src/packets.py tests/test_packets.py && git commit -m "Add evidence packets and renderer"`.

### Task 3: Runner

**Files:**
- Create: `src/run.py`, `src/pricing.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: Task 2 `render`, `build_prompt`, `item_ids`, `special_item_id`, `TICKERS`, `CONDITIONS`.
- Produces: `RunSpec = namedtuple("RunSpec", "model ticker condition repeat")`; `plan_runs(models, tickers, conditions, repeats) -> List[RunSpec]` (round-robin: repeat outermost, then model, ticker, condition); `completed_keys(path: str) -> Set[Tuple[str,str,str,int]]`; `parse_output(text: str, valid_ids: List[str]) -> Tuple[Optional[dict], bool, List[str]]` returning `(parsed, parse_ok, unknown_ids)`; `execute(client, spec: RunSpec, max_attempts: int = 3) -> dict` (the JSONL record); `cost_usd(model: str, usage) -> float` in `src/pricing.py`; CLI `python3 -m src.run --models ... --tickers ... --conditions ... --repeats N --out runs/runs.jsonl --concurrency K`.
- JSONL record keys: `run_id, model, model_reported, ticker, condition, direction, repeat, special_item_id, packet_sha256, prompt_sha256, ts, duration_s, stop_reason, raw_text, parsed, parse_ok, unknown_ids, attempts, input_tokens, output_tokens, cost_usd, error`.

- [ ] **Step 1: Write failing tests** in `tests/test_run.py` using a `FakeClient` whose `messages.create` returns a canned object (`content=[obj(type="text", text=...)]`, `stop_reason`, `model`, `usage.input_tokens/output_tokens`):
  - `test_plan_runs_round_robin`: with 2 models, 1 ticker, 2 conditions, 2 repeats, the first four specs all have `repeat == 0`.
  - `test_parse_rejects_out_of_range`: rating 6 → `parse_ok False`; 2 drivers → False; confidence 1.2 → False; valid → True.
  - `test_parse_flags_unknown_ids`: evidence id `"NBIS-99"` not in `valid_ids` → appears in `unknown_ids`, `parse_ok` stays True.
  - `test_refusal_logged_as_failed`: fake returns `stop_reason="refusal"`, no text → record has `parse_ok False`, `error == "refusal"`, `attempts == 3`.
  - `test_resume_skips_completed`: write a JSONL with one completed record; `completed_keys` returns its key; `plan_runs` minus completed excludes it.
  - `test_cost_usd`: opus 1000 in / 500 out → `0.004 + 0.01 = 0.014`.
- [ ] **Step 2: Run tests** → FAIL.
- [ ] **Step 3: Implement `src/pricing.py`** (`PRICES = {"claude-opus-5-5": (4.0, 20.0), ...}` per 1M tokens) and `src/run.py`: `.env` loader (parse `KEY=VALUE` lines, set `os.environ` if unset); `anthropic.Anthropic()` client with `max_retries=5`; request per spec §4; `execute` retries on `parse_ok False` or API exception up to `max_attempts`, catching `anthropic.APIStatusError` and `anthropic.APIConnectionError` separately; `ThreadPoolExecutor(concurrency)`; append each record with a lock, `flush` after every line; `--dry-run` prints the first prompt and exits.
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Smoke the real API once** (key must exist): `python3 -m src.run --models claude-haiku-5-5 --tickers NBIS --conditions C0 --repeats 1 --out runs/smoke.jsonl` → one line with `parse_ok true`. If the API rejects `minItems`/`maxItems`, remove them from `prompts/schema.json` (local check in `parse_output` already covers it) and note it in the Task 1 commit's follow-up.
- [ ] **Step 6: Commit** `git add src/run.py src/pricing.py tests/test_run.py runs/smoke.jsonl && git commit -m "Add resumable API runner"`.

### Task 4: Pilot

**Files:**
- Create: `runs/pilot.jsonl`, `docs/pilot-notes.md`

- [ ] **Step 1: Run** `python3 -m src.run --models claude-sonnet-5-5 --tickers NBIS --repeats 3 --out runs/pilot.jsonl --concurrency 4` (12 runs).
- [ ] **Step 2: Check** with a one-off `python3 -c` over the JSONL: all 12 `parse_ok`; C0 ratings neither all identical nor spanning 1–5; `special_item_id` cited in ≥1 of 3 runs for C1 and C2; mean cost per run × 480 < $40. Write the four numbers to `docs/pilot-notes.md`.
- [ ] **Step 3: Decide**: if any check fails, adjust `prompts/` or `packets/` (not `METRICS.md`), re-run the pilot into a fresh file, and record why in `docs/pilot-notes.md`. Pilot runs count toward the full run only if nothing in `prompts/` or `packets/` changed afterwards (compare `prompt_sha256`).
- [ ] **Step 4: Commit** `git add runs/pilot.jsonl docs/pilot-notes.md && git commit -m "Pilot: 12 Sonnet runs on NBIS"`.

### Task 5: Metrics

**Files:**
- Create: `src/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `pairwise_disagreement(a: Sequence[int], b: Sequence[int], same: bool = False) -> float` (NaN if fewer than 2 pairs); `cell_metrics(ratings: Dict[str, List[int]], attributed: Dict[str, List[bool]], direction: int) -> Dict[str, float]` with keys `N, F1, F2, F3, E1, E2, E3, S1, S2, S3, A1, A2, A3, n0, n1, n2, n3` where `S_k` is signed `mean(R_k) − mean(R_0)` multiplied by `direction` for k in {1,2} (so positive = moved the expected way) and raw for k=3; `bootstrap_cell(ratings, attributed, direction, n_boot=2000, seed=0) -> Dict[str, Tuple[float,float,float]]` (point, lo, hi) resampling runs within each condition; `diagnosis(E1: float, E2: float, eps: float = 0.05) -> str` returning `"resistant-responsive" | "insensitive" | "credulous" | "mixed"`.

- [ ] **Step 1: Write failing tests**:
  - `test_pairwise_same`: `pairwise_disagreement([1,1,2],[1,1,2], same=True) == 2/3`.
  - `test_pairwise_cross`: `pairwise_disagreement([3,3],[4,4]) == 1.0`; `([3,4],[3,4]) == 0.5`.
  - `test_cell_metrics_hand`: R0=[3,3,3,4], R1=[4,4,4,4], direction +1 → `N == 0.5`, `F1 == 0.75`, `E1 == 0.25`, `S1 == 0.75`.
  - `test_direction_flips_sign`: same data with direction −1 → `S1 == -0.75`.
  - `test_attribution`: attributed C1 `[True, False, True, True]` → `A1 == 0.75`.
  - `test_metrics_with_tiny_cells`: R0=[3], R1=[] → all of `N, E1, S1` are NaN, no exception.
  - `test_bootstrap_bounds`: with the hand data, `bootstrap_cell` returns `lo <= point <= hi` for `E1`, and is reproducible with the same seed.
  - `test_diagnosis`: (0.0, 0.4) → resistant-responsive; (0.0, 0.0) → insensitive; (0.4, 0.4) → credulous.
- [ ] **Step 2: Run tests** → FAIL.
- [ ] **Step 3: Implement `src/metrics.py`** with numpy; `diagnosis`: E1 ≤ eps and E2 > eps → resistant-responsive; both ≤ eps → insensitive; E1 > eps and |E1 − E2| ≤ eps → credulous; else mixed.
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** `git add src/metrics.py tests/test_metrics.py && git commit -m "Add pre-registered metrics with bootstrap"`.

### Task 6: Full run

**Files:**
- Create/append: `runs/runs.jsonl`

- [ ] **Step 1: Seed** `runs/runs.jsonl` with the pilot lines if the pilot counts (Task 4 step 3); otherwise start empty.
- [ ] **Step 2: Run** `python3 -m src.run --repeats 10 --out runs/runs.jsonl --concurrency 6` (defaults: all models, tickers, conditions). Restart the same command if it stops; it resumes.
- [ ] **Step 3: Verify** with a one-off count: 480 lines, per-cell counts of `parse_ok` ≥ 8; total `cost_usd` summed; list any cell below 8 valid runs.
- [ ] **Step 4: Commit** `git add runs/runs.jsonl && git commit -m "Full run: 480 API calls"`.

### Task 7: Analysis, tables and figures

**Files:**
- Create: `src/analyze.py`, `tests/test_analyze.py`, `tests/fixtures/runs_small.jsonl`
- Outputs: `results/metrics.csv`, `results/summary.csv`, `results/table1.tex`, `results/fig1_ratings.pdf`, `results/fig2_excess.pdf`, `results/cost.txt`, `results/cutoffs.json`

**Interfaces:**
- Consumes: Task 5 `cell_metrics`, `bootstrap_cell`, `diagnosis`; Task 2 `DIRECTION`, `special_item_id`.
- Produces: `load_runs(path) -> pandas.DataFrame` (one row per run, columns from the JSONL record plus `rating`, `confidence`, `attributed: bool`); `per_cell(df) -> DataFrame` (rows = model × ticker, columns = metric point/lo/hi); `pooled(df_cells) -> DataFrame` (rows = model; mean over tickers of each metric, run-level bootstrap pooled by averaging ticker resamples); `write_table1(pooled_df, path)` (booktabs LaTeX, metrics N, F1, E1, E2, E3, S1, S2, A1, A2, with CIs in small text); `fig_ratings(df, path)`, `fig_excess(pooled_df, path)`; CLI `python3 -m src.analyze --runs runs/runs.jsonl --out results/`.

- [ ] **Step 1: Write `tests/fixtures/runs_small.jsonl`**: 2 models × 2 tickers × 4 conditions × 3 repeats = 48 synthetic records (ratings chosen by hand so NBIS/opus has E1 ≈ 0, E2 > 0), plus one cell with all runs `parse_ok false`.
- [ ] **Step 2: Write failing tests**: `test_load_runs_shape` (47 valid rows, `attributed` bool); `test_per_cell_has_expected_values` (the hand-chosen cell's `E1` point ≈ 0 within 1e-9, `E2` > 0); `test_analyze_handles_missing_cell` (the all-failed cell yields NaN and the CLI still writes all seven output files); `test_table1_is_valid_latex` (file contains `\toprule` and one row per model).
- [ ] **Step 3: Run tests** → FAIL.
- [ ] **Step 4: Implement `src/analyze.py`**; figures with matplotlib, `rcParams["pdf.fonttype"]=42`, width 6.3 in for full text width, colorblind-safe palette (Okabe–Ito); fig 1 = small multiples (rows models, cols tickers) of rating histograms per condition (four offset bars per rating value); fig 2 = per model, E1/E2/E3 points with CI whiskers and hollow markers for F1/F2/F3; `cost.txt` = total and per-model USD; `cutoffs.json` = for each model, the stated training cutoff obtained by one extra API call per model asking "What is your training data cutoff? Answer with a month and year only." (logged, 3 calls).
- [ ] **Step 5: Run tests** → PASS; then run the CLI on `runs/runs.jsonl` and eyeball both PDFs (open them with Read).
- [ ] **Step 6: Commit** `git add src/analyze.py tests results && git commit -m "Add analysis, tables and figures"`.

### Task 8: Paper

**Files:**
- Create: `paper/` (clone of the AES template: `aes.sty`, `aes-author-year.bst`, `fonts/`, `assets/`), `paper/main.tex`, `paper/references.bib`, `paper/build.sh`, `tools/tectonic` (gitignored), `paper/main.pdf`

- [ ] **Step 1: Fetch tooling**: `git clone --depth 1 https://github.com/agent-evalscience/AES-Latex-Template.git paper` then `rm -rf paper/.git paper/main-review.tex paper/main.pdf paper/main-review.pdf`; download `https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-aarch64-apple-darwin.tar.gz` into `tools/`, untar, `chmod +x tools/tectonic`.
- [ ] **Step 2: Write `paper/build.sh`**: `cd paper && ../tools/tectonic -X compile main.tex` (Tectonic runs XeTeX and BibTeX automatically; the `.bst` and `fonts/` sit beside `main.tex`). Run it on the untouched template `main.tex` once → `main.pdf` exists. If fontspec cannot find the bundled Open Sans files, point `aes.sty`'s font path at `fonts/` (check `docs/LAYOUT.md`) rather than installing fonts system-wide.
- [ ] **Step 3: Build `paper/references.bib`**: for each arXiv ID in spec §9 fetch `https://export.arxiv.org/api/query?id_list=<ids comma-separated>` and write `@article` entries with the returned title, authors, year, `eprint`, `archivePrefix={arXiv}`; add `@misc` entries with `howpublished={\url{...}}` and `note={Accessed 2026-10-23}` for Vals Finance Agent v2, the SemiAnalysis index post, and the three company filings used. IDs the API does not resolve are dropped from the paper, not guessed; list them in `docs/pilot-notes.md`.
- [ ] **Step 4: Write `paper/main.tex`** per spec §9: front matter (`\title[Noise-Calibrated Conclusion Robustness]{...}`, `\aesauthor[affil={1},corresponding]{Zixin Audrey Cai}`, `\aesaffiliation{1}{Peira Labs}`, `\aesemails{\aesemail{zixincai@umich.edu}}`, `\aeskeywords{Agent evaluation; Robustness; Finance; Reliability}`), abstract ≤ 200 words, sections 1–7, `\input{../results/table1.tex}` inside a `table` float, `\includegraphics` of the two figures, appendix with the system prompt, the question, and one `aeslongtable` listing every packet item (id, date, role, synthetic flag, first sentence). Every number in the text is copied from `results/summary.csv` or `results/cost.txt`; the text names the three diagnosis labels per model from `results/summary.csv`.
- [ ] **Step 5: Build and check length**: `sh paper/build.sh`; count pages before the References heading with `python3 -c` over `pdftotext`-free means: use `tools/tectonic`'s log page count plus a visual check of `main.pdf` with Read; the body must end on page 4–6. Trim or expand text accordingly; appendix pages do not count.
- [ ] **Step 6: Fresh-eyes review**: dispatch a subagent to read `paper/main.pdf` as a demanding reviewer for the Diagnose track against the venue's four criteria (clear evaluation target and bounded claims; sound methods and reporting; findings relevant to evaluation research; scope). Fix what it finds that the logs support; do not add claims.
- [ ] **Step 7: Commit** `git add paper && git commit -m "Paper: AES short paper draft"` (`paper/main.pdf` is committed so the submission artifact is versioned).

### Task 9: Release notes

**Files:**
- Create: `README.md`; modify `.gitignore` if needed.

- [ ] **Step 1: Write `README.md`**: one-paragraph description, synthetic-content notice (Global Constraints), reproduce commands (install, tests, pilot, full run, analyze, build paper), file map from spec §10, the as-of date and price-source note, and the cost actually spent from `results/cost.txt`.
- [ ] **Step 2: Run the whole test suite** `python3 -m pytest -q` → all pass.
- [ ] **Step 3: Commit** `git add README.md && git commit -m "Add README and release notes"`.
- [ ] **Step 4: Report** to the author: path to `paper/main.pdf`, the OpenReview link, the per-model diagnosis line, total spend, and anything that deviated from the spec.
