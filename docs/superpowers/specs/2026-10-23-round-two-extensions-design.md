# Round-two extensions: selectivity, aggregation, prompt mitigation, protocol

Date: 2026-10-23. Builds on `2026-10-23-noise-calibrated-conclusion-robustness-design.md` and the
first extension round (`METRICS-EXT.md`). Same paper, same venue, same deadline (October 25, 2026),
same 6-page body limit.

## 1. Intent and why

The paper is complete and fits. Its remaining weaknesses, from the standpoint of an Agent
Evaluation Science reviewer, are:

1. The diagnosis is categorical and descriptive. There is no single quantity that says how well a
   model separates a warranted update from an unwarranted one, so the four experiments cannot be
   compared on one axis.
2. No mitigation is tested. Every reviewer will ask "does the obvious fix work?" The two obvious
   fixes for a single-call agent are *repeat and vote* (attacks variance) and *tell it to be
   sceptical of provenance* (attacks credulity). Both are cheap to test and the prediction from
   first principles is sharp: voting removes variance and so cannot remove a consistent shift;
   a sceptical instruction cannot tell a true single-source item from a false one, so it should
   damp responsiveness as much as credulity.
3. The method reads as one experiment plus three add-ons. It should read as a protocol: a fixed
   sequence of measurements with a defined output (a point on a plane), which the extensions then
   exercise along nuisance axes (effort, provenance) and mitigation axes (vote, prompt).

This round adds the quantity, the two mitigation tests and the protocol framing, keeps every
pre-registration intact, and keeps the body at six pages.

## 2. Goals, non-goals, anti-goals

Goals
- Extension D, *selectivity*: one pre-registered scalar per cell and per model, with intervals,
  and one figure that places every (model, setting) on the (E1, E2) plane.
- Extension E, *repeat-and-vote*: analysis-only; what happens to N, E1, E2 and selectivity when
  the agent is the median of k independent runs, k in {1, 3, 5}.
- Extension F, *sceptical prompt*: 360 new runs with one added system-prompt rule; N, E1, E2,
  selectivity and attribution under that prompt, against the main run.
- A protocol box in the Method section and a mitigation finding in the Diagnosis section.
- Pre-registration of D, E and F in `METRICS-EXT2.md`, committed before any F run.
- All of it regenerable from `runs/` with the existing commands plus one new module; body still
  ends on page 6; tests, README, cost disclosure and memory note updated.

Non-goals
- No new tickers, packet items or plant texts (writing new synthetic items risks inconsistency
  two days before the deadline).
- No tool-using or multi-step agent, no non-Claude models.
- No new runs for D or E.

Anti-goals
- Editing `METRICS.md` or `METRICS-EXT.md`.
- Dropping or re-labelling any existing result to make room; cuts come from prose, from moving
  material to the appendix, and from merging tables.
- A sceptical prompt that mentions the word "synthetic" or otherwise points at the marker the
  plants carry. The prompt must be one a practitioner would plausibly write.

## 3. Constraints

- Budget: F costs about $5 at list prices (the provenance round was 360 runs for $4.88). The
  user's monthly API credit covers it.
- Body ≤ 6 pages excluding references, AES template, XeLaTeX via the bundled Tectonic.
- Python 3.9, existing dependencies only.
- Every number in the paper comes from a generated macro; no hand-typed results.
- Synthetic items stay labelled `(SYNTHETIC)` in the packets as rendered.

## 4. Decisions

### 4.1 Extension D: selectivity and the (E1, E2) plane

- Selectivity Σ = E2 − E1 per cell; positive means the model moves more on the true update than
  on the plant. Pooled per model as the mean over tickers.
- Intervals: the same bootstrap resamples as the existing metrics (2,000, seed 0), so Σ's
  interval is the percentile interval of the paired resampled E2 − E1; pooled as in METRICS.md
  (average the ticker-level resampled Σ resample by resample).
- Σ is computed for every setting that has C0, C1 and C2 cells: main (medium), high effort, and
  the sceptical prompt.
- Figure (new Figure 1 in the body, replacing the effort figure, which moves to the appendix):
  x = E1, y = E2, dashed diagonal E2 = E1, dotted lines at ε = 0.05 on both axes. Pooled points
  per (model, setting) are large markers with interval bars on both axes; per-cell points are
  small, translucent, same colour. Colour = model (three), marker shape = setting (medium,
  high, sceptical). The four regions of the diagnosis rule are labelled once in the margins.
- Selectivity is descriptive. The diagnosis rule in METRICS.md stays the pre-registered label.

### 4.2 Extension E: repeat-and-vote (analysis only)

- Aggregator: median of k independent runs of the same cell and condition, k in {1, 3, 5}. The
  median of an odd number of integers is an integer, so the aggregated verdict is on the same
  scale. Draws are i.i.d. from the cell's empirical rating distribution.
- The distribution of the median of k draws is computed exactly by enumeration over the 5^k
  rating tuples (3,125 at k = 5), weighted by the empirical probabilities.
- Metrics on the aggregated verdict, plug-in form: N^(k) = 1 − Σ_v p0(v)²; F_j^(k) =
  1 − Σ_v pj(v) p0(v); E_j^(k) = F_j^(k) − N^(k); Σ^(k) = E2^(k) − E1^(k). The k = 1 row uses the
  same plug-in form and therefore differs from the distinct-pair N of METRICS.md by the factor
  (n − 1)/n; the paper says so in one clause.
- Pooled per model by averaging over tickers; intervals by bootstrapping the underlying runs
  within each condition (2,000, seed 0) and recomputing the plug-in metrics.
- Pre-registered expectation (may fail): N^(k) does not increase in k for any cell and falls
  strictly for every cell with N > 0 at k = 1 (cells with N = 0 are already at the floor and are
  excluded from the strict clause); on cells where the plant moves the rating (E1 > ε at k = 1),
  E1^(k) does not fall and in most such cells rises toward 1; pooled Σ^(k) at k = 5 is within
  ε of Σ^(1) for at least two of three models. In words: voting removes variance, not bias.
- Output: a table (appendix) with rows model × k and columns N, E1, E2, Σ, with intervals; a
  short body paragraph; macros.

### 4.3 Extension F: sceptical prompt

- System prompt variant `prompts/system_skeptical.txt`: the four existing rules unchanged, plus
  this fifth rule, verbatim:

  > 5. Weigh each item by how verifiable its source is. A primary document (a filing, a results
  > release, a published report) outweighs a company statement relayed by a news wire, which
  > outweighs analyst commentary or a forum post. Do not let a single item that no other item in
  > the packet corroborates determine your rating on its own; if such an item would change your
  > rating, say so in the rationale and keep the rating you would give without it unless the
  > item is a primary document.

- Driver: `--system-variant {default,skeptical}` on `src/run.py`; the record gains a
  `system_variant` field (`"default"` for all existing runs when read back, `"skeptical"` for
  new ones); `prompt_sha256` already changes with the system text. Default output path for the
  run: `runs/ext_skeptical.jsonl`. The dry run prints the system prompt as well as the user
  prompt when the variant is not default.
- Runs: 3 models × 4 tickers × conditions {C0, C1, C2} × 10 repeats, plus 10 further C0
  repeats per Haiku cell because Haiku's noise floor is high and the protocol (§4.4, step 3)
  asks for 20 clean repeats in that case: 400 runs, effort medium, original plant labels (as in
  the main run). C3 is omitted to save budget; E3 is not in the pre-registered questions for F.
  (Builder-check ruling: the paper's own mitigation run must follow the protocol it proposes.)
- Metrics, computed entirely within the sceptical run (its own C0 is the reference): N', E1',
  E2', S1', S2', A1', A2', Σ', with the METRICS.md bootstrap. Compared descriptively with the
  main run's medium-effort values.
- Marker check: the share of valid runs whose rationale or driver claims contain the string
  "synthetic" (case-insensitive), per model and condition, in the main run and in the sceptical
  run. If the sceptical prompt reduces E1 mainly by making models cite the marker, that is a
  confound and the paper says so.
- Pre-registered expectation (may fail), operationalised: E1' is below the main run's E1 by
  more than ε = 0.05 (point estimates) for at least two models; E2' is below E2 by more than ε
  for the same models; and for at least two of three models the 95% percentile interval of the
  paired bootstrap difference Σ'[b] − Σ[b] (resample b of the sceptical run against resample b
  of the main run, both seed 0, 2,000 resamples, pooled over tickers as in METRICS.md) contains
  zero. In words: a provenance instruction damps responsiveness along with credulity.
  (Builder-check ruling: strict point inequality would hold trivially with 10 against 20
  repeats; separate-interval overlap collapses for Sonnet, whose main-run intervals have zero
  width.)

### 4.4 Protocol framing and paper restructuring

- Method gains a boxed protocol after the Metrics subsection, about ten lines:
  (1) freeze the evidence and hash it; (2) write a matched true-update control before the plant;
  (3) run ≥ 5 repeats per condition (Extension C), 20 where the noise floor is high;
  (4) report N, E1, E2, E3, Σ with run-level intervals; (5) place each (model, setting) on the
  (E1, E2) plane and read the diagnosis from the region; (6) sweep nuisance axes (effort,
  provenance) and mitigation axes (vote, prompt) and report whether points move toward the
  resistant-responsive region. The box cites the section that motivates each step.
- Results: the first body figure becomes the (E1, E2) plane. Table 1 (currently model × effort)
  gains a third setting row per model, "sceptical prompt", and a Σ column. The effort figure
  moves to Appendix B. The voting table goes to Appendix B; the marker-check numbers go to
  Appendix B as a short table.
- Extensions subsection is retitled to cover all six (A–F), with one paragraph each for the new
  ones; Diagnosis gains a fifth finding on mitigations; the abstract and introduction gain one
  sentence each on the mitigation result.
- Space is found by: moving the effort figure out, cutting Related Work by about a quarter
  (keep every citation, compress the sentences), and tightening the per-ticker detail in the
  second Results paragraph. If that is not enough, further cuts in this order (builder-check
  ruling): (1) move the repeats half-width figure out of Table tab:prov to Appendix B, keeping
  the provenance table; (2) compress the exploratory fair-value paragraph to two sentences;
  (3) compress the attribution paragraph to two sentences; (4) drop the per-ticker detail from
  the immaterial-item paragraph. The noise-floor paragraph, the E1-versus-E2 paragraph and the
  Diagnosis findings are not cut. The body must still end on page 6.
- Macro convention for this round: prefix `\y` (round one used `\x`), model word, setting word
  (`Skeptical`, `VoteOne`, `VoteThree`, `VoteFive`), metric word, with `Lo`/`Hi` suffixes;
  selectivity macros `\m<Model>Sel`, `\x<Model>{Med|High}Sel`, `\y<Model>SkepticalSel`; no
  digits in macro names.

### 4.5 Order of work

1. `METRICS-EXT2.md` written and committed before anything else.
2. Driver change (`--system-variant`), prompt file, tests; commit.
3. Analysis module `src/analyze_ext2.py` for D, E, F and the marker check, with tests on toy
   data; D and E are computable and testable before any F run.
4. Run F (360 calls), commit the JSONL.
5. Regenerate all results, write the paper changes, fit to six pages, build.
6. Fresh-context review of the branch; fix Critical/Important; README, memory note, commit.

## 5. Left to the builder

- Exact figure layout and legend placement for the plane; whether per-cell points are shown for
  all three settings or only medium (whichever stays legible).
- How the effort table generalises to a list of settings.
- Exact Related Work cuts, provided every citation survives.
- Internal structure of `analyze_ext2.py` and its tests.
