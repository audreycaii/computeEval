from src import packets as P

SEPT30_CLOSE = {"NBIS": "235.88", "CRWV": "87.12", "IREN": "40.88", "NVDA": "228.38"}


def _by_role(ticker):
    out = {}
    for it in P.load_packet(ticker)["items"]:
        out.setdefault(it["role"], []).append(it)
    return out


def test_roles_per_ticker():
    for t in P.TICKERS:
        r = _by_role(t)
        assert len(r["true_update"]) == 1, t
        assert len(r["false_plant"]) == 1, t
        assert len(r["noise"]) == 1, t
        assert 8 <= len(r["base"]) <= 12, (t, len(r["base"]))


def test_special_items_only_in_their_condition():
    for t in P.TICKERS:
        r = _by_role(t)
        fp, tu, nz = r["false_plant"][0]["id"], r["true_update"][0]["id"], r["noise"][0]["id"]
        assert P.special_item_id(t, "C0") is None
        assert P.special_item_id(t, "C1") == fp
        assert P.special_item_id(t, "C2") == tu
        assert P.special_item_id(t, "C3") == nz
        c0 = set(P.item_ids(t, "C0"))
        assert not ({fp, tu, nz} & c0)
        assert set(P.item_ids(t, "C1")) == c0 | {fp}
        assert set(P.item_ids(t, "C2")) == c0 | {tu}
        assert set(P.item_ids(t, "C3")) == c0 | {nz}


def test_render_deterministic():
    for t in P.TICKERS:
        shas = set()
        for c in P.CONDITIONS:
            text1, sha1 = P.render(t, c)
            text2, sha2 = P.render(t, c)
            assert text1 == text2 and sha1 == sha2
            shas.add(sha1)
        assert len(shas) == 4


def test_synthetic_flags():
    for t in P.TICKERS:
        for it in P.load_packet(t)["items"]:
            if it["role"] in ("false_plant", "noise"):
                assert it["synthetic"] is True, it["id"]
            else:
                assert it["synthetic"] is False, it["id"]
                assert it["url"], it["id"]


def test_price_anchor_present():
    for t in P.TICKERS:
        text, _ = P.render(t, "C0")
        assert SEPT30_CLOSE[t] in text, t


def test_items_sorted_by_date():
    for t in P.TICKERS:
        for c in P.CONDITIONS:
            text, _ = P.render(t, c)
            items = {it["id"]: it for it in P.load_packet(t)["items"]}
            positions = [(text.index("[" + i + "]"), items[i]["date"]) for i in P.item_ids(t, c)]
            dates_in_text_order = [d for _, d in sorted(positions)]
            assert dates_in_text_order == sorted(dates_in_text_order), (t, c)


def test_build_prompt_fills_placeholders():
    p = P.build_prompt("NBIS", "C0")
    assert "{" not in p.split("Evidence packet")[0]
    assert "Nebius" in p and "2026-10-01" in p and "[NBIS-" in p


def test_direction_table():
    assert P.DIRECTION == {"NBIS": 1, "CRWV": -1, "IREN": -1, "NVDA": 1}
