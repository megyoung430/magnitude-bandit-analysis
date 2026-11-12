"""Assembling the code-map data from extracted records, descriptions and overviews."""

import pytest

from src.code_map.assemble import (
    Overviews,
    build_map,
    coverage,
    definition_entry,
    overviews_from_toml,
    package_of,
    parse_descriptions,
)
from src.code_map.extract import module_record, notebook_record

SOURCE = '''"""Toy analysis."""


def fit(trials, n_starts: int = 5) -> dict:
    """Fit one mouse.

    Args:
        trials (pd.DataFrame): One row per trial.
        seed (int): Passed through ``**kwargs``.

    Returns:
        dict: The fit.
    """
    return {}


def _save(path) -> None:
    pass


def _undescribed(x):
    return x
'''

OVERVIEWS = Overviews(
    intro="The repo.",
    sections=(("src/", "The library."),),
    packages={"src/behavior_analysis": "Pure analyses.", "src/agents": "Agents."},
    modules={},
    notebooks={"scripts/nb.ipynb": "A notebook."},
)


def toy_module():
    return module_record("src/behavior_analysis/toy.py", SOURCE)


def test_inputs_join_signature_types_with_documented_descriptions():
    fit = definition_entry(toy_module().defs[0], "src/behavior_analysis/toy.py", {})
    assert fit["i"] == [
        ["trials", "pd.DataFrame", "One row per trial."],
        ["n_starts", "int", ""],
        ["seed", "int", "Passed through ``**kwargs``."],
    ]
    assert fit["o"] == "dict: The fit." and fit["d"] == "Fit one mouse."


def test_a_written_description_fills_a_missing_docstring_and_is_flagged():
    descriptions = {"src/behavior_analysis/toy.py::_save": "Writes the fit."}
    save = definition_entry(toy_module().defs[1], "src/behavior_analysis/toy.py", descriptions)
    assert save["d"] == "Writes the fit." and save["u"] == 1
    assert save["o"] == "None (acts by side effect)"


def test_coverage_lists_undescribed_definitions_and_stale_descriptions():
    descriptions = {
        "src/behavior_analysis/toy.py::_save": "Writes the fit.",
        "src/gone.py::old": "No longer exists.",
    }
    missing, stale = coverage([toy_module()], descriptions)
    assert missing == ["src/behavior_analysis/toy.py::_undescribed"]
    assert stale == ["src/gone.py::old"]


def test_package_is_the_first_two_path_parts_or_the_top_level():
    assert package_of("src/agents/q.py") == "src/agents"
    assert package_of("src/settings.py") == "src (top level)"
    assert package_of("scripts/full_pipelines/run.py") == "scripts/full_pipelines"


def test_build_map_orders_packages_as_the_overviews_list_them_and_appends_unknown_ones():
    modules = [
        module_record("src/agents/q.py", '"""Q."""\n'),
        module_record("src/new_pkg/x.py", '"""X."""\n'),
        toy_module(),
    ]
    notebook = notebook_record("scripts/nb.ipynb", {"cells": []})
    data = build_map(modules, [notebook], OVERVIEWS, {}, "main @ abc1234")
    assert [p["name"] for p in data["pkgs"]] == [
        "src/behavior_analysis", "src/agents", "src/new_pkg", "scripts (top level)",
    ]  # fmt: skip
    assert data["pkgs"][2]["about"] == ""
    assert data["meta"] == "main @ abc1234"
    assert data["pkgs"][3]["mods"][0]["a"] == "A notebook."


def test_module_summary_prefers_the_written_overview_over_the_docstring():
    overviews = OVERVIEWS._replace(modules={"src/agents/q.py": "Written."})
    data = build_map([module_record("src/agents/q.py", '"""Q."""\n')], [], overviews, {}, "")
    assert data["pkgs"][0]["mods"][0]["a"] == "Written."


def test_descriptions_are_tab_separated_key_and_text_with_comments_allowed():
    text = "# path::qualname<TAB>description\nsrc/a.py::f\tDoes f.\n\nsrc/a.py::g\tDoes g.\n"
    assert parse_descriptions(text) == {"src/a.py::f": "Does f.", "src/a.py::g": "Does g."}


@pytest.mark.parametrize(
    "text, message",
    [
        ("src/a.py::f\tOne.\nsrc/a.py::f\tTwo.\n", "twice"),
        ("src/a.py::f Does f.\n", "line 1"),
        ("src/a.py\tNo qualname.\n", "line 1"),
    ],
)
def test_bad_description_lines_fail_loudly(text, message):
    with pytest.raises(ValueError, match=message):
        parse_descriptions(text)


def test_overviews_from_toml_keeps_order_and_requires_the_intro():
    parsed = {
        "intro": "Hi.",
        "section": [{"name": "src/", "text": "Lib."}],
        "package": {"src/b": "B.", "src/a": "A."},
    }
    ov = overviews_from_toml(parsed)
    assert list(ov.packages) == ["src/b", "src/a"] and ov.sections == (("src/", "Lib."),)
    assert ov.modules == {} and ov.notebooks == {}
    with pytest.raises(ValueError, match="intro"):
        overviews_from_toml({"package": {}})
