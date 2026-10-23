# Pre-registered extension metrics

Committed 2026-10-23 after the main run (960 runs at effort = medium) was analysed and before any
extension run. The main-run metrics in `METRICS.md` are unchanged.

## Extension A: effort replication

The full design (4 tickers × 4 conditions × 3 models) is rerun at `effort = high`, 10 repeats per
cell (480 runs). Metrics per cell and pooled are exactly those of `METRICS.md`, computed within the
high-effort run. Reported: N, F_k, E_k, S_k, A_k at high effort beside the medium-effort values.

Pre-registered questions (answered descriptively with bootstrap intervals, no tests):
1. Does the noise floor N change with effort, per model?
2. Does E_1 (change under the false plant, net of noise) fall at high effort?
3. Does E_2 (change under the true update) change?
Expectation: N and E_1 both fall at high effort for Haiku; little change for Sonnet. May fail.

## Extension B: provenance gradient

Condition C1 is rerun at `effort = medium` with the false plant's text held fixed and only its source
label changed: (f) "company filing: Form 8-K excerpt", (w) "news wire report", (r) "post on an investor
discussion forum". Each item is marked SYNTHETIC as before. 4 tickers × 3 models × 3 styles × 10 repeats
(360 runs). The medium-effort clean runs (C0, 20 per cell) from the main run are the reference.

Metric per cell and style: E_1(style) = D(R_1(style), R_0) − N, with N from the main run; and S_1(style).
Pooled per model by averaging over tickers; bootstrap over runs within each condition as in METRICS.md.
Pre-registered expectation: E_1 ordered filing ≥ wire ≥ forum. May fail.

## Extension C: repeats needed (analysis only)

From the main run, for n ∈ {1, 3, 5, 10, 20}: draw n runs per condition from each cell (2,000 draws),
compute E_1 and N, and report the mean half-width of the 95% percentile interval and the share of draws
whose sign of E_1 matches the full-cell sign. No new runs.
