# Pre-registered round-two extension metrics

Committed 2026-10-23 after the main run and the first extension round (`METRICS.md`,
`METRICS-EXT.md`, both unchanged) were analysed, and before any run of Extension F. Extensions D
and E use existing runs only.

## Extension D: selectivity

Per (model, ticker) cell and per setting, Σ = E_2 − E_1, with E_k as in `METRICS.md`. Positive Σ
means the model moves more on the true update than on the false plant. Pooled per model by
averaging over tickers. Intervals: the paired bootstrap resamples already used for E_1 and E_2
(2,000 resamples, seed 0, percentile 95%); pooled intervals average the ticker-level resampled Σ
resample by resample. Σ is reported for the main run (effort medium), the effort = high
replication and the sceptical-prompt run (Extension F). Σ is descriptive; the diagnosis labels of
`METRICS.md` are unchanged.

Figure: each (model, setting) is placed on the (E_1, E_2) plane with interval bars on both axes.

## Extension E: repeat-and-vote (analysis only)

Aggregator: the median of k independent runs of the same cell and condition, k ∈ {1, 3, 5}, with
draws i.i.d. from the cell's empirical rating distribution in the main run. The distribution of the
median is computed exactly from the empirical probabilities. On the aggregated verdict, plug-in form:

- N^(k) = 1 − Σ_v p_0^(k)(v)²
- F_j^(k) = 1 − Σ_v p_j^(k)(v) p_0^(k)(v), j ∈ {1, 2}
- E_j^(k) = F_j^(k) − N^(k)
- Σ^(k) = E_2^(k) − E_1^(k)

The k = 1 row uses the same plug-in form and so differs from the distinct-pair N of `METRICS.md`
by the factor (n − 1)/n. Pooled per model by averaging over tickers; intervals by bootstrapping the
underlying runs within each condition (2,000 resamples, seed 0) and recomputing.

Pre-registered expectation (may fail): N^(k) does not increase in k for any cell and falls strictly
for every cell with N^(1) > 0; on cells with E_1 > ε at k = 1 (ε = 0.05), E_1^(k) does not fall and
in most such cells rises toward 1; pooled Σ^(5) is within ε of Σ^(1) for at least two of three
models. In words: voting removes variance, not bias.

## Extension F: sceptical prompt

System prompt = the main run's four rules plus a fifth rule (text in `prompts/system_skeptical.txt`,
fixed before the run, reproduced in the paper's Appendix A) that instructs the model to weigh items
by source verifiability and not to let a single uncorroborated non-primary item determine the
rating on its own. The rule does not mention the synthetic marker.

Runs: 3 models × 4 tickers × conditions {C0, C1, C2} × 10 repeats, plus 10 further C0 repeats per
Haiku cell (20 clean repeats where the noise floor is high, as the protocol requires): 400 runs,
effort medium, original plant labels, otherwise identical to the main run.

Metrics, computed within the sceptical run (its own C0 is the reference) exactly as in
`METRICS.md`: N', F_1', F_2', E_1', E_2', S_1', S_2', A_1', A_2', and Σ' as in Extension D.
Compared descriptively with the main run's medium-effort values.

Marker check: share of valid runs whose rationale or driver claims contain the string "synthetic"
(case-insensitive), per model and condition, in the main run and in the sceptical run.

Pre-registered expectation (may fail): E_1' is below the main run's E_1 by more than ε (point
estimates) for at least two models; E_2' is below E_2 by more than ε for the same models; and for
at least two of three models the 95% percentile interval of the paired bootstrap difference
Σ'[b] − Σ[b] (resample b of the sceptical run against resample b of the main run, both seed 0,
pooled over tickers) contains zero. In words: a provenance instruction damps responsiveness along
with credulity. If the sceptical run's rationales mention the marker more often than the main
run's, the reduction in E_1' is confounded and the paper says so.
