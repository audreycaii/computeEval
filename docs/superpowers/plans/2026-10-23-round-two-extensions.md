# Round-Two Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selectivity (D), repeat-and-vote (E) and a sceptical-prompt mitigation run (F) to the paper, with a protocol box, keeping the body at six pages.

**Architecture:** One new analysis module `src/analyze_ext2.py` reusing `analyze.per_cell`/`pooled` and `metrics`; a `--system-variant` switch in the driver; paper edits in `paper/main.tex`. All numbers flow through generated macros (`results/numbers_ext2.tex`, prefix `\y`).

**Tech Stack:** Python 3.9, numpy, pandas, matplotlib, pytest; anthropic SDK 0.125.0; Tectonic.

**Spec:** `docs/superpowers/specs/2026-10-23-round-two-extensions-design.md`; pre-registration `METRICS-EXT2.md` (frozen, commit f18afad).

## Global Constraints

- Never edit `METRICS.md`, `METRICS-EXT.md`, `METRICS-EXT2.md`, `prompts/system_skeptical.txt`.
- Macro names contain no digits. Round-two prefix `\y`.
- ε = 0.05; bootstrap 2,000 resamples, seed 0, percentile 95%.
- Body ends on page 6; references start on page 7.
- Existing results must regenerate byte-identically (`python3 -m src.analyze`, `python3 -m src.analyze_ext`).

## Review Focus

1. A JSONL with mixed `system_variant` values must be refused by the driver, not silently merged.
2. `load_runs` on old records (no `system_variant`, no `mentions_synthetic`) must default to `"default"`/False.
3. Median-of-k pmf must equal brute-force enumeration for every k used, including degenerate single-value cells.
4. Paired Σ difference must pair resample b with resample b, not re-draw.
5. Settings table must print `--` for E3 under the sceptical setting (no C3) rather than NaN or crash.

---

### Task 1: Driver system variant

**Files:** Modify `src/run.py`, `src/analyze.py` (`load_runs`); Test `tests/test_run.py`, `tests/test_analyze.py`.

**Interfaces:**
- Produces: `run.SYSTEM_VARIANTS = {"default": "system.txt", "skeptical": "system_skeptical.txt"}`; `run.system_prompt(variant: str = "default") -> str`; `build_request(..., system_variant: str = "default")`; `execute(..., system_variant: str = "default")` writes `rec["system_variant"]`; `run.check_variant(path, variant)` raises `ValueError` if the file's first record has a different variant; CLI `--system-variant`, default `--out` becomes `runs/ext_skeptical.jsonl` when variant is skeptical and `--out` not given; `analyze.load_runs` gains columns `system_variant` (default `"default"`) and `mentions_synthetic` (bool over rationale + driver claims, case-insensitive).

- [ ] Tests: `test_system_prompt_variant_adds_rule_five`, `test_execute_records_system_variant`, `test_check_variant_refuses_mixed_file`, `test_load_runs_defaults_variant_and_marker` (fixture rows with and without the fields; one rationale containing "Synthetic").
- [ ] Run, watch fail; implement; run, pass; commit.

### Task 2: analyze_ext2 D and E (no new runs)

**Files:** Create `src/analyze_ext2.py`; Test `tests/test_analyze_ext2.py`.

**Interfaces:**
- `add_selectivity(cells: pd.DataFrame) -> pd.DataFrame`: adds `Sel`, `Sel_lo`, `Sel_hi` and `attrs["samples"][(m,t)]["Sel"]`.
- `pooled_with_sel(cells) -> pd.DataFrame`: `analyze.pooled` plus pooled `Sel` and interval from averaged samples.
- `setting_frame(path: str, setting: str, n_boot=2000, seed=0) -> Tuple[cells, pooled]` with a leading `setting` column on both.
- `median_pmf(p: np.ndarray, k: int) -> np.ndarray` over ratings 1..5 (index 0..4), via binomial CDF.
- `vote(main_path, ks=(1,3,5), n_boot=2000, seed=0) -> Tuple[cells_df, pooled_df]` columns `k, model, [ticker], N, E1, E2, Sel` + `_lo/_hi` on pooled.
- `write_settings_table(pooled_settings, path)`, `write_vote_table(vote_pooled, path)`, `fig_plane(pooled_settings, cells_settings, path)`.

- [ ] Tests: `test_median_pmf_matches_enumeration` (k ∈ {1,3,5}, random p and a one-hot p), `test_vote_noise_floor_falls_with_k` (toy cell with noisy C0), `test_selectivity_is_e2_minus_e1_paired` (Sel == E2 − E1 pointwise and samples paired), `test_settings_table_dashes_missing_e3`.
- [ ] Implement, pass, commit. Run `setting_frame` on the real main and high runs as a smoke check (no files written yet).

### Task 3: Extension F run

- [ ] `python3 -m src.run --system-variant skeptical --conditions C0 C1 C2 --repeats 10 --concurrency 6` (360 runs) then `python3 -m src.run --system-variant skeptical --models claude-haiku-5-5 --conditions C0 --repeats 20 --concurrency 6` (+40). Expected: 400 records, all `parse_ok`, cost about $5.
- [ ] Commit `runs/ext_skeptical.jsonl`.

### Task 4: analyze_ext2 F outputs and macros

**Interfaces:**
- `skeptical_comparison(main_cells, skep_cells) -> pd.DataFrame`: per model, `E1_main, E1_skep, dE1, E2_main, E2_skep, dE2, Sel_main, Sel_skep, dSel, dSel_lo, dSel_hi` (paired samples).
- `marker_check(paths: Dict[str,str]) -> pd.DataFrame`: `setting, model, condition, share`.
- `write_ext2_numbers(pooled_settings, vote_pooled, comp, marker, path)`; `write_marker_table`.
- `main()` writes `results/ext2_settings.csv`, `ext2_settings_table.tex`, `ext2_vote.csv`, `ext2_vote_table.tex`, `ext2_skeptical.csv`, `ext2_marker.csv`, `ext2_marker_table.tex`, `fig5_plane.pdf`, `numbers_ext2.tex`.

- [ ] Tests: `test_skeptical_comparison_pairs_samples`, `test_marker_check_counts_case_insensitive`, `test_ext2_numbers_have_no_digits_in_names`.
- [ ] Implement, pass, run `python3 -m src.analyze_ext2`, commit results.

### Task 5: Paper

- [ ] Add `\input{../results/numbers_ext2.tex}`; protocol box after Metrics; Table 1 → `ext2_settings_table.tex`; Figure 1 → `fig5_plane.pdf`; effort figure, vote table, marker table → Appendix B; paragraphs D, E, F in Extensions; finding 5 in Diagnosis; abstract/intro sentences; rule 5 in Appendix A; Limitations mention of marker check.
- [ ] Build; body ends page 6 (apply the spec's cut order as needed); no undefined macros in `main.log`; commit.

### Task 6: Verification, review, docs

- [ ] Full test suite; staleness check (regenerate all results and `git status` clean apart from PDF timestamp).
- [ ] Fresh-context reviewer on the paper diff and the analysis module; fix Critical/Important.
- [ ] README (layout, reproduce, cost), memory note; sync `paper` branch; commit.
