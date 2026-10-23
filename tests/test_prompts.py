import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_schema():
    with open(os.path.join(ROOT, "prompts", "schema.json")) as f:
        return json.load(f)


def test_schema_shape():
    s = _load_schema()
    assert set(s["required"]) == {"rating", "confidence", "drivers", "rationale"}
    assert s["additionalProperties"] is False
    assert s["properties"]["rating"]["minimum"] == 1
    assert s["properties"]["rating"]["maximum"] == 5
    assert s["properties"]["confidence"]["minimum"] == 0
    assert s["properties"]["confidence"]["maximum"] == 1
    driver = s["properties"]["drivers"]["items"]
    assert set(driver["required"]) == {"claim", "evidence_ids"}
    assert driver["additionalProperties"] is False


def test_question_placeholders():
    with open(os.path.join(ROOT, "prompts", "question.txt")) as f:
        q = f.read()
    for ph in ("{company}", "{ticker}", "{as_of_date}", "{packet}"):
        assert ph in q


def test_system_prompt_core_instructions():
    with open(os.path.join(ROOT, "prompts", "system.txt")) as f:
        s = f.read().lower()
    assert "only" in s and "packet" in s
    assert "admissible" in s
    assert "json" in s
