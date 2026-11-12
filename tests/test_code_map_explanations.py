"""Plain-language explanations: parsing explanations.md and placing them on the page."""

import pytest

from src.code_map.assemble import Overviews, build_map, stale_explanations
from src.code_map.explanations import parse_explanations
from src.code_map.extract import module_record

TEXT = """# Plain-language explanations

Anything before the first entry is a preamble and is ignored.

## src/a.py

A file primer.

It can have two paragraphs.

## src/a.py::fit

Fits a curve.

- first step
- second step
"""


def test_entries_are_keyed_by_their_heading_and_keep_paragraphs_and_lists():
    found = parse_explanations(TEXT)
    assert list(found) == ["src/a.py", "src/a.py::fit"]
    assert found["src/a.py"] == "A file primer.\n\nIt can have two paragraphs."
    assert found["src/a.py::fit"] == "Fits a curve.\n\n- first step\n- second step"


def test_backticks_around_the_key_are_allowed():
    assert parse_explanations("## `src/a.py::fit`\n\nFits.\n") == {"src/a.py::fit": "Fits."}


@pytest.mark.parametrize(
    "text, message",
    [
        ("## src/a.py\n\nOne.\n\n## src/a.py\n\nTwo.\n", "twice"),
        ("## src/a.py\n\n", "empty"),
        ("## not a path\n\nText.\n", "line 1"),
    ],
)
def test_bad_entries_fail_loudly(text, message):
    with pytest.raises(ValueError, match=message):
        parse_explanations(text)


SOURCE = '''"""Toy."""


def fit(x):
    """Fit."""
    return x
'''

OVERVIEWS = Overviews("Intro.", (), {"src (top level)": "Top."}, {}, {})


def test_explanations_land_on_the_file_and_the_function():
    modules = [module_record("src/a.py", SOURCE)]
    explanations = {"src/a.py": "Primer.", "src/a.py::fit": "Fits."}
    data = build_map(modules, [], OVERVIEWS, {}, "m", explanations)
    mod = data["pkgs"][0]["mods"][0]
    assert mod["w"] == "Primer." and mod["fn"][0]["w"] == "Fits."


def test_explanations_for_missing_files_or_functions_are_stale():
    modules = [module_record("src/a.py", SOURCE)]
    keys = {"src/a.py": "", "src/a.py::fit": "", "src/a.py::gone": "", "src/b.py": ""}
    assert stale_explanations(modules, [], keys) == ["src/a.py::gone", "src/b.py"]
