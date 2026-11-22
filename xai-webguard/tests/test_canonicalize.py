import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_webguard.canonicalize import canonicalize


def test_percent_decoding():
    c = canonicalize("%3Cscript%3E")
    assert c.text == "<script>"
    assert c.layers >= 1


def test_entity_and_unicode():
    assert canonicalize("&lt;img&gt;").text == "<img>"
    assert "alert" in canonicalize(r"alert(1)").text


def test_offset_map_points_back_to_raw():
    raw = "id=1%20OR%201=1"
    c = canonicalize(raw)
    i = c.text.lower().find("or")
    rs, re_ = c.raw_span(i, i + 2)
    # the mapped raw slice should contain the encoded 'OR'
    assert "OR" in raw[rs:re_] or "or" in raw[rs:re_].lower()


def test_double_encoding_layers():
    # %2533 -> %33 -> 3
    c = canonicalize("%2533")
    assert c.layers >= 2


def test_empty_input():
    c = canonicalize("")
    assert c.text == "" and c.src == []
