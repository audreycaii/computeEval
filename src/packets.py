"""Evidence packets: load, select items per condition, render to prompt text."""
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKET_DIR = os.path.join(ROOT, "packets")
PROMPT_DIR = os.path.join(ROOT, "prompts")

TICKERS = ["NBIS", "CRWV", "IREN", "NVDA"]
CONDITIONS = ["C0", "C1", "C2", "C3"]
DIRECTION = {"NBIS": 1, "CRWV": -1, "IREN": -1, "NVDA": 1}
COMPANY = {
    "NBIS": "Nebius Group",
    "CRWV": "CoreWeave",
    "IREN": "IREN Limited",
    "NVDA": "NVIDIA Corporation",
}
AS_OF = "2026-10-01"
# Extension B: the false plant's text is held fixed and only its (source_type, source) label changes.
PLANT_STYLES = {
    "filing": ("company filing", "Form 8-K excerpt filed with the SEC (SYNTHETIC)"),
    "wire": ("news wire", "News wire report (SYNTHETIC)"),
    "forum": ("forum", "Post on an investor discussion forum (SYNTHETIC)"),
}
ROLE_FOR_CONDITION = {"C0": None, "C1": "false_plant", "C2": "true_update", "C3": "noise"}
# Round three (METRICS-EXT3.md): the synthetic marker stays in the data and the paper; the rendered
# packet drops it unless marker_visible is set (which reproduces the earlier, labelled runs).
MARKER = " (SYNTHETIC)"


def load_packet(ticker: str) -> Dict:
    with open(os.path.join(PACKET_DIR, ticker + ".json"), encoding="utf-8") as f:
        return json.load(f)


def _items(ticker: str, condition: str, plant_style: Optional[str] = None) -> List[Dict]:
    role = ROLE_FOR_CONDITION[condition]
    items = [dict(it) for it in load_packet(ticker)["items"] if it["role"] == "base" or it["role"] == role]
    if plant_style is not None:
        stype, source = PLANT_STYLES[plant_style]  # KeyError for an unknown style
        for it in items:
            if it["role"] == "false_plant":
                it["source_type"], it["source"] = stype, source
    return sorted(items, key=lambda it: (it["date"], it["id"]))


def special_item_id(ticker: str, condition: str) -> Optional[str]:
    role = ROLE_FOR_CONDITION[condition]
    if role is None:
        return None
    for it in load_packet(ticker)["items"]:
        if it["role"] == role:
            return it["id"]
    raise ValueError("no %s item for %s" % (role, ticker))


def item_ids(ticker: str, condition: str) -> List[str]:
    return [it["id"] for it in _items(ticker, condition)]


def render(ticker: str, condition: str, plant_style: Optional[str] = None,
           marker_visible: bool = False) -> Tuple[str, str]:
    items = _items(ticker, condition, plant_style)
    lines = ["Evidence packet for %s (%s), as of %s. %d items." % (COMPANY[ticker], ticker, AS_OF, len(items)), ""]
    for it in items:
        source = it["source"] if marker_visible else it["source"].replace(MARKER, "")
        lines.append("[%s] (%s, %s, %s) %s" % (it["id"], it["date"], it["source_type"], source, it["text"]))
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt(ticker: str, condition: str, plant_style: Optional[str] = None,
                 marker_visible: bool = False) -> str:
    with open(os.path.join(PROMPT_DIR, "question.txt"), encoding="utf-8") as f:
        template = f.read()
    packet_text, _ = render(ticker, condition, plant_style, marker_visible)
    return template.format(company=COMPANY[ticker], ticker=ticker, as_of_date=AS_OF, packet=packet_text)
