# Pilot notes (Task 4), 2026-10-23

Three pilot rounds, each Sonnet 5.5 × NBIS × 4 conditions × 3 repeats (12 runs, ~$0.12 each round).

| Round | File | Change before it | Ratings C0/C1/C2/C3 | Special item cited (C1,C2,C3) | Note |
|---|---|---|---|---|---|
| 1 | `runs/pilot_v1_discarded.jsonl` | — | 3,3,3 / 2,2,2 / 3,3,3 / 3,3,3 | 3/3, 3/3, 0/3 | Rationales flagged a market-cap conflict in the NBIS packet ($52B from Yahoo shares vs ">$65B" from a secondary outlet). Packet item NBIS-08 trimmed to the price-history fact only. Discarded. |
| 2 | `runs/pilot_v2_discarded.jsonl` | NBIS-08 fix | 3,3,3 / 3,3,3 / 3,3,3 / 3,3,3 | 3/3, 3/3, 0/3 | Rating collapsed to 3 in every cell: the 5-point scale has no resolution for this model at medium effort. Added an exploratory continuous output `fair_value_usd` to the schema and question. Discarded. |
| 3 | `runs/pilot.jsonl` | `fair_value_usd` added | 3,3,3 / 3,3,3 / 3,3,3 / 3,3,3 | 3/3, 3/3, 0/3 | Fair value: C0 230,230,230 / C1 235,235,235 / C2 235,235,235 / C3 230,230,230. Kept; counts toward the full run. |

Checks against the plan:
- All outputs parse: yes (36/36 across rounds). One round-2 run hit the 8,000-token output cap on its first attempt and succeeded on retry.
- Clean ratings not all identical: **no**, all 3. Ruling: this is a measurement, not a defect; the prompt is not tuned to induce variance. A zero noise floor is reported as such.
- Special item cited in C1/C2: yes, 100%; never cited in C3.
- Cost: mean $0.0100 per Sonnet run → 480 runs ≈ $8 (Opus ≈ 2×, Haiku ≈ 0.1×). Well under the $40 ceiling, so repeats are raised from 10 to 20 (960 runs, ≈ $16–20) for resolution on small effects.

Pre-registered metrics (`METRICS.md`) are unchanged. `fair_value_usd` is exploratory: the paper reports its shift (percent of the C0 mean) alongside the rating metrics and labels it as added after the pilot.
