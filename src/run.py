"""Resumable driver: one Claude API call per run, one JSONL line per run."""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

from src import packets as P
from src.pricing import cost_usd

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
EFFORT = "medium"
MAX_TOKENS = 8000
# Extension F: the sceptical variant adds one rule to the system prompt (METRICS-EXT2.md).
SYSTEM_VARIANTS = {"default": "system.txt", "skeptical": "system_skeptical.txt"}
RunSpec = namedtuple("RunSpec", "model ticker condition repeat")

_SCHEMA = None


def schema() -> Dict:
    global _SCHEMA
    if _SCHEMA is None:
        with open(os.path.join(P.PROMPT_DIR, "schema.json"), encoding="utf-8") as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


_UNSUPPORTED = {"minimum", "maximum", "minItems", "maxItems"}


def api_schema() -> Dict:
    """The schema sent to the API: the documented schema minus keywords the API rejects.
    The stripped constraints are enforced locally in parse_output."""
    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in _UNSUPPORTED}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node
    return strip(schema())


def system_prompt(variant: str = "default") -> str:
    with open(os.path.join(P.PROMPT_DIR, SYSTEM_VARIANTS[variant]), encoding="utf-8") as f:  # KeyError for unknown
        return f.read()


def load_env(path: str = os.path.join(P.ROOT, ".env")) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def plan_runs(models: List[str], tickers: List[str], conditions: List[str], repeats: int) -> List[RunSpec]:
    return [RunSpec(m, t, c, r) for r in range(repeats) for m in models for t in tickers for c in conditions]


def completed_keys(path: str) -> Set[Tuple[str, str, str, int]]:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["model"], rec["ticker"], rec["condition"], rec["repeat"]))
    return done


def check_variant(path: str, variant: str) -> None:
    """Refuse to append runs of one system variant to a file holding another. Records written before
    the field existed are default-prompt runs."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                found = json.loads(line).get("system_variant", "default")
                if found != variant:
                    raise ValueError("%s holds system_variant=%r, refusing to append %r runs" % (path, found, variant))
    return None


def parse_output(text: str, valid_ids: List[str]) -> Tuple[Optional[Dict], bool, List[str]]:
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None, False, []
    try:
        rating, conf, drivers = data["rating"], data["confidence"], data["drivers"]
        if not (isinstance(rating, int) and not isinstance(rating, bool) and 1 <= rating <= 5):
            return data, False, []
        if not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
            return data, False, []
        fv = data.get("fair_value_usd")
        if not (isinstance(fv, (int, float)) and fv > 0):
            return data, False, []
        if not (isinstance(drivers, list) and len(drivers) == 3):
            return data, False, []
        unknown = []
        for d in drivers:
            for eid in d["evidence_ids"]:
                if eid not in valid_ids and eid not in unknown:
                    unknown.append(eid)
        if not isinstance(data.get("rationale"), str):
            return data, False, unknown
        return data, True, unknown
    except (KeyError, TypeError):
        return data, False, []


def build_request(model: str, ticker: str, condition: str, effort: str = EFFORT,
                  plant_style: Optional[str] = None, system_variant: str = "default") -> Dict:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS if effort != "high" else 16000,
        "system": system_prompt(system_variant),
        "messages": [{"role": "user", "content": P.build_prompt(ticker, condition, plant_style)}],
        "output_config": {"format": {"type": "json_schema", "schema": api_schema()}, "effort": effort},
    }


def _text_of(response) -> Optional[str]:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return None


def execute(client, spec: RunSpec, max_attempts: int = 3, effort: str = EFFORT,
            plant_style: Optional[str] = None, system_variant: str = "default") -> Dict:
    req = build_request(spec.model, spec.ticker, spec.condition, effort=effort, plant_style=plant_style,
                        system_variant=system_variant)
    _, packet_sha = P.render(spec.ticker, spec.condition, plant_style)
    prompt_sha = hashlib.sha256((req["system"] + "\n" + req["messages"][0]["content"]).encode("utf-8")).hexdigest()
    valid_ids = P.item_ids(spec.ticker, spec.condition)
    rec = {
        "run_id": uuid.uuid4().hex, "model": spec.model, "model_reported": None, "ticker": spec.ticker,
        "condition": spec.condition, "direction": P.DIRECTION[spec.ticker], "repeat": spec.repeat,
        "effort": effort, "plant_style": plant_style, "system_variant": system_variant,
        "special_item_id": P.special_item_id(spec.ticker, spec.condition), "packet_sha256": packet_sha,
        "prompt_sha256": prompt_sha, "ts": None, "duration_s": None, "stop_reason": None, "raw_text": None,
        "parsed": None, "parse_ok": False, "unknown_ids": [], "attempts": 0, "input_tokens": 0,
        "output_tokens": 0, "cost_usd": 0.0, "error": None,
    }
    for attempt in range(1, max_attempts + 1):
        rec["attempts"] = attempt
        rec["parsed"], rec["parse_ok"], rec["unknown_ids"], rec["raw_text"] = None, False, [], None
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        t0 = time.time()
        try:
            resp = client.messages.create(**req)
        except Exception as e:  # anthropic.APIStatusError / APIConnectionError and anything else
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            rec["duration_s"] = round(time.time() - t0, 2)
            time.sleep(min(60, 5 * attempt))
            continue
        rec["duration_s"] = round(time.time() - t0, 2)
        rec["model_reported"] = getattr(resp, "model", None)
        rec["stop_reason"] = getattr(resp, "stop_reason", None)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            rec["input_tokens"] += usage.input_tokens
            rec["output_tokens"] += usage.output_tokens
            rec["cost_usd"] += cost_usd(spec.model, usage)
        text = _text_of(resp)
        rec["raw_text"] = text
        if rec["stop_reason"] == "refusal":
            rec["error"] = "refusal"
            continue
        if rec["stop_reason"] == "max_tokens":
            rec["error"] = "max_tokens"
            continue
        parsed, ok, unknown = parse_output(text, valid_ids)
        rec["parsed"], rec["parse_ok"], rec["unknown_ids"] = parsed, ok, unknown
        if ok:
            rec["error"] = None
            break
        rec["error"] = "parse"
    rec["cost_usd"] = round(rec["cost_usd"], 6)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tickers", nargs="+", default=P.TICKERS)
    ap.add_argument("--conditions", nargs="+", default=P.CONDITIONS)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default=None, help="default runs/runs.jsonl, or runs/ext_skeptical.jsonl for --system-variant skeptical")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--effort", default=EFFORT, choices=["low", "medium", "high"])
    ap.add_argument("--plant-style", default=None, choices=sorted(P.PLANT_STYLES))
    ap.add_argument("--system-variant", default="default", choices=sorted(SYSTEM_VARIANTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(P.ROOT, "runs", "runs.jsonl" if args.system_variant == "default" else "ext_%s.jsonl" % args.system_variant)

    specs = plan_runs(args.models, args.tickers, args.conditions, args.repeats)
    if args.dry_run:
        s = specs[0]
        req = build_request(s.model, s.ticker, s.condition, args.effort, args.plant_style, args.system_variant)
        if args.system_variant != "default":
            print(req["system"] + "\n----")
        print(req["messages"][0]["content"])
        return 0
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (put it in .env)", file=sys.stderr)
        return 2
    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    check_variant(args.out, args.system_variant)
    done = completed_keys(args.out)
    todo = [s for s in specs if (s.model, s.ticker, s.condition, s.repeat) not in done]
    print("planned %d, done %d, to run %d" % (len(specs), len(done), len(todo)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "cost": 0.0}

    def work(spec):
        rec = execute(client, spec, effort=args.effort, plant_style=args.plant_style, system_variant=args.system_variant)
        with lock:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            counters["ok" if rec["parse_ok"] else "fail"] += 1
            counters["cost"] += rec["cost_usd"]
            n = counters["ok"] + counters["fail"]
            print("[%d/%d] %s %s %s r%d ok=%s rating=%s $%.4f" % (
                n, len(todo), spec.model.split("-")[1], spec.ticker, spec.condition, spec.repeat,
                rec["parse_ok"], (rec["parsed"] or {}).get("rating"), counters["cost"]), flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, todo))
    print("done: ok=%d fail=%d cost=$%.4f" % (counters["ok"], counters["fail"], counters["cost"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
