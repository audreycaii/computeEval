#!/bin/sh
# Round three: all four batches with the marker hidden (METRICS-EXT3.md). Each line resumable.
cd "$(dirname "$0")/.." || exit 1
(python3 -m src.run --repeats 20 --out runs/runs.jsonl --concurrency 5 > runs/full_run.log 2>&1) &
(python3 -m src.run --effort high --repeats 10 --out runs/ext_effort_high.jsonl --concurrency 3 > runs/ext_effort_high.log 2>&1) &
(for st in filing wire forum; do python3 -m src.run --conditions C1 --plant-style $st --repeats 10 --concurrency 3 --out runs/ext_provenance_$st.jsonl; done > runs/ext_provenance.log 2>&1) &
(python3 -m src.run --system-variant skeptical --conditions C0 C1 C2 --repeats 10 --concurrency 3 && python3 -m src.run --system-variant skeptical --models claude-haiku-5-5 --conditions C0 --repeats 20 --concurrency 3) > runs/ext_skeptical.log 2>&1 &
wait
echo ALL DONE
