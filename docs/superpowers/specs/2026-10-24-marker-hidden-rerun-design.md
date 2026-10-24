# Round three: hide the synthetic marker from the model and rerun

Date: 2026-10-24. Same paper, venue, deadline (October 25, 2026) and 6-page body.

## Intent

The author ruled that the "(SYNTHETIC)" suffix belongs in the released data and the paper, not in
the prompt the agent reads. Every run so far showed it to the model. All four run batches are
repeated with the suffix stripped at render time; the earlier runs become a marker-visibility
comparison (Extension G) in the appendix. Pre-registration: `METRICS-EXT3.md`, committed first.

## Decisions

- `packets.render(ticker, condition, plant_style=None, marker_visible=False)`: strips
  " (SYNTHETIC)" from `source` unless `marker_visible`. `PLANT_STYLES` labels lose the suffix in
  the same way (strip at render, keep the constant so the visible rendering reproduces the old
  packet hashes byte for byte).
- Driver: `--marker-visible` flag (default off) for reproducing the old runs; record field
  `marker_visible`; `check_variant` also refuses to append runs whose `marker_visible` differs
  from the file's (missing field = visible). Default output paths unchanged, so the old files move
  to `runs/labelled/` first (`git mv`), logs with them.
- `analyze.load_runs` gains `marker_visible` (default True for old records).
- Analysis modules keep their default paths and therefore pick up the new runs. A new function in
  `analyze_ext2` (or a small `analyze_ext3`) computes the Extension G table and macros
  (prefix `\z`, no digits) from `runs/runs.jsonl` against `runs/labelled/runs.jsonl`.
- Paper: Conditions paragraph says the marker is in the data and appendix only, not shown to the
  model; the marker check becomes a construction check plus Extension G; Extension F's confound
  paragraph is rewritten from the hidden-run numbers; every qualitative claim is re-read against
  the regenerated macros; README cost and reproduce sections updated; body still ends on page 6.

## Non-goals

No new items, tickers, metrics or models. No edits to the three earlier METRICS files or the
sceptical prompt.
