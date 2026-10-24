import json
import os
from types import SimpleNamespace as NS

import pytest

from src import run as R
from src.pricing import cost_usd


def _valid_output(rating=4, drivers=None, confidence=0.7, fair_value=210.0):
    if drivers is None:
        drivers = [{"claim": "c%d" % i, "evidence_ids": ["NBIS-04"]} for i in range(3)]
    return json.dumps({"rating": rating, "fair_value_usd": fair_value, "confidence": confidence, "drivers": drivers, "rationale": "r"})


class FakeMessages:
    def __init__(self, text=None, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, 0

    def create(self, **kwargs):
        self.calls += 1
        content = [NS(type="text", text=self.text)] if self.text is not None else []
        return NS(content=content, stop_reason=self.stop_reason, model=kwargs["model"],
                  usage=NS(input_tokens=1000, output_tokens=500), stop_details=None)


class FakeClient:
    def __init__(self, text=None, stop_reason="end_turn"):
        self.messages = FakeMessages(text, stop_reason)


def test_plan_runs_round_robin():
    specs = R.plan_runs(["m1", "m2"], ["NBIS"], ["C0", "C1"], 2)
    assert len(specs) == 8
    assert [s.repeat for s in specs[:4]] == [0, 0, 0, 0]
    assert [s.repeat for s in specs[4:]] == [1, 1, 1, 1]


def test_parse_rejects_out_of_range():
    ids = ["NBIS-04"]
    assert R.parse_output(_valid_output(rating=6), ids)[1] is False
    two = [{"claim": "a", "evidence_ids": ["NBIS-04"]}] * 2
    assert R.parse_output(_valid_output(drivers=two), ids)[1] is False
    assert R.parse_output(_valid_output(confidence=1.2), ids)[1] is False
    assert R.parse_output("not json", ids)[1] is False
    assert R.parse_output(_valid_output(fair_value=0), ids)[1] is False
    assert R.parse_output(_valid_output(fair_value=-5), ids)[1] is False
    parsed, ok, unknown = R.parse_output(_valid_output(), ids)
    assert ok is True and parsed["rating"] == 4 and parsed["fair_value_usd"] == 210.0 and unknown == []


def test_parse_flags_unknown_ids():
    drivers = [{"claim": "a", "evidence_ids": ["NBIS-99"]}] + [{"claim": "b", "evidence_ids": ["NBIS-04"]}] * 2
    parsed, ok, unknown = R.parse_output(_valid_output(drivers=drivers), ["NBIS-04"])
    assert ok is True
    assert unknown == ["NBIS-99"]


def test_refusal_logged_as_failed():
    client = FakeClient(text=None, stop_reason="refusal")
    rec = R.execute(client, R.RunSpec("claude-haiku-5-5", "NBIS", "C0", 0), max_attempts=3)
    assert rec["parse_ok"] is False
    assert rec["error"] == "refusal"
    assert rec["attempts"] == 3
    assert client.messages.calls == 3


def test_execute_valid_record_fields():
    client = FakeClient(text=_valid_output())
    rec = R.execute(client, R.RunSpec("claude-haiku-5-5", "NBIS", "C1", 2))
    assert rec["parse_ok"] is True and rec["attempts"] == 1
    assert rec["ticker"] == "NBIS" and rec["condition"] == "C1" and rec["repeat"] == 2
    assert rec["direction"] == 1 and rec["special_item_id"] == "NBIS-21"
    assert rec["input_tokens"] == 1000 and rec["output_tokens"] == 500
    assert rec["cost_usd"] == pytest.approx(0.0001 + 0.00025)
    assert len(rec["packet_sha256"]) == 64 and len(rec["prompt_sha256"]) == 64
    assert rec["model_reported"] == "claude-haiku-5-5"


def test_resume_skips_completed(tmp_path):
    path = str(tmp_path / "runs.jsonl")
    rec = {"model": "m1", "ticker": "NBIS", "condition": "C0", "repeat": 0, "parse_ok": True}
    with open(path, "w") as f:
        f.write(json.dumps(rec) + "\n")
    done = R.completed_keys(path)
    assert done == {("m1", "NBIS", "C0", 0)}
    specs = [s for s in R.plan_runs(["m1"], ["NBIS"], ["C0"], 2) if (s.model, s.ticker, s.condition, s.repeat) not in done]
    assert [s.repeat for s in specs] == [1]


def test_completed_keys_missing_file(tmp_path):
    assert R.completed_keys(str(tmp_path / "nope.jsonl")) == set()


def test_cost_usd():
    assert cost_usd("claude-opus-5-5", NS(input_tokens=1000, output_tokens=500)) == pytest.approx(0.014)
    assert cost_usd("claude-sonnet-5-5", NS(input_tokens=1000, output_tokens=500)) == pytest.approx(0.007)


def test_api_schema_strips_unsupported_keywords():
    s = R.api_schema()
    assert "minimum" not in s["properties"]["rating"] and "maximum" not in s["properties"]["rating"]
    assert "minimum" not in s["properties"]["confidence"]
    assert "minItems" not in s["properties"]["drivers"] and "maxItems" not in s["properties"]["drivers"]
    assert s["properties"]["rating"]["type"] == "integer"
    assert R.schema()["properties"]["rating"]["minimum"] == 1  # original untouched
    assert R.build_request("claude-haiku-5-5", "NBIS", "C0")["output_config"]["format"]["schema"] == s


def test_parse_rejects_bool_rating():
    out = json.loads(_valid_output())
    out["rating"] = True
    assert R.parse_output(json.dumps(out), ["NBIS-04"])[1] is False


def test_record_reset_between_attempts():
    class Flaky:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return NS(content=[NS(type="text", text=_valid_output(rating=9))], stop_reason="end_turn",
                          model=kwargs["model"], usage=NS(input_tokens=1, output_tokens=1), stop_details=None)
            return NS(content=[], stop_reason="refusal", model=kwargs["model"],
                      usage=NS(input_tokens=1, output_tokens=1), stop_details=None)

    client = NS(messages=Flaky())
    rec = R.execute(client, R.RunSpec("claude-haiku-5-5", "NBIS", "C0", 0), max_attempts=2)
    assert rec["parse_ok"] is False and rec["error"] == "refusal" and rec["parsed"] is None


def test_build_request_effort_and_plant_style():
    req = R.build_request("claude-haiku-5-5", "NBIS", "C1", effort="high")
    assert req["output_config"]["effort"] == "high"
    req2 = R.build_request("claude-haiku-5-5", "NBIS", "C1", plant_style="forum")
    assert "investor discussion forum" in req2["messages"][0]["content"]


def test_execute_records_effort_and_style():
    client = FakeClient(text=_valid_output())
    rec = R.execute(client, R.RunSpec("claude-haiku-5-5", "NBIS", "C1", 0), effort="high", plant_style="wire")
    assert rec["effort"] == "high" and rec["plant_style"] == "wire"
    rec0 = R.execute(client, R.RunSpec("claude-haiku-5-5", "NBIS", "C1", 0))
    assert rec0["effort"] == "medium" and rec0["plant_style"] is None


def test_system_prompt_variant_adds_rule_five():
    base, skep = R.system_prompt(), R.system_prompt("skeptical")
    assert skep.startswith(base.rstrip("\n"))
    assert "5. Weigh each item by how verifiable its source is" in skep and "5." not in base
    assert "synthetic" not in skep.lower()
    with pytest.raises(KeyError):
        R.system_prompt("nope")


def test_execute_records_system_variant():
    client = FakeClient(text=_valid_output())
    rec = R.execute(client, R.RunSpec("claude-haiku-5-5", "NBIS", "C1", 0))
    assert rec["system_variant"] == "default"
    rec2 = R.execute(FakeClient(text=_valid_output()), R.RunSpec("claude-haiku-5-5", "NBIS", "C1", 0), system_variant="skeptical")
    assert rec2["system_variant"] == "skeptical"
    assert rec2["prompt_sha256"] != rec["prompt_sha256"] and rec2["packet_sha256"] == rec["packet_sha256"]
    assert R.build_request("claude-haiku-5-5", "NBIS", "C1", system_variant="skeptical")["system"] == R.system_prompt("skeptical")


def test_check_variant_refuses_mixed_file(tmp_path):
    path = str(tmp_path / "runs.jsonl")
    assert R.check_variant(path, "skeptical") is None  # missing file is fine
    with open(path, "w") as f:
        f.write(json.dumps({"model": "m", "ticker": "NBIS", "condition": "C0", "repeat": 0, "system_variant": "skeptical"}) + "\n")
    assert R.check_variant(path, "skeptical") is None
    with pytest.raises(ValueError):
        R.check_variant(path, "default")
    legacy = str(tmp_path / "old.jsonl")
    with open(legacy, "w") as f:
        f.write(json.dumps({"model": "m", "ticker": "NBIS", "condition": "C0", "repeat": 0}) + "\n")
    assert R.check_variant(legacy, "default", marker_visible=True) is None  # records without the field are default runs
    with pytest.raises(ValueError):
        R.check_variant(legacy, "skeptical")
    mixed = str(tmp_path / "mixed.jsonl")  # a later record with another variant is caught too
    with open(mixed, "w") as f:
        f.write(json.dumps({"model": "m", "ticker": "NBIS", "condition": "C0", "repeat": 0, "system_variant": "default"}) + "\n")
        f.write(json.dumps({"model": "m", "ticker": "NBIS", "condition": "C0", "repeat": 1, "system_variant": "skeptical"}) + "\n")
    with pytest.raises(ValueError):
        R.check_variant(mixed, "default")


def test_execute_records_marker_hidden_by_default():
    rec = R.execute(FakeClient(text=_valid_output()), R.RunSpec("claude-haiku-5-5", "NBIS", "C1", 0))
    assert rec["marker_visible"] is False
    assert "SYNTHETIC" not in R.build_request("claude-haiku-5-5", "NBIS", "C1")["messages"][0]["content"]
    rec_v = R.execute(FakeClient(text=_valid_output()), R.RunSpec("claude-haiku-5-5", "NBIS", "C1", 0), marker_visible=True)
    assert rec_v["marker_visible"] is True
    assert rec_v["packet_sha256"] != rec["packet_sha256"] and rec_v["prompt_sha256"] != rec["prompt_sha256"]
    assert "(SYNTHETIC)" in R.build_request("claude-haiku-5-5", "NBIS", "C1", marker_visible=True)["messages"][0]["content"]


def test_check_variant_refuses_mixed_marker(tmp_path):
    legacy = str(tmp_path / "old.jsonl")
    with open(legacy, "w") as f:
        f.write(json.dumps({"model": "m", "ticker": "NBIS", "condition": "C0", "repeat": 0}) + "\n")
    assert R.check_variant(legacy, "default", marker_visible=True) is None  # records without the field showed the marker
    with pytest.raises(ValueError):
        R.check_variant(legacy, "default")  # default: marker hidden
    new = str(tmp_path / "new.jsonl")
    with open(new, "w") as f:
        f.write(json.dumps({"model": "m", "ticker": "NBIS", "condition": "C0", "repeat": 0, "marker_visible": False}) + "\n")
    assert R.check_variant(new, "default") is None
    with pytest.raises(ValueError):
        R.check_variant(new, "default", marker_visible=True)
