"""Extracting definitions from Python source and blocks from notebooks."""

import pytest

from src.code_map.extract import module_record, notebook_record

SOURCE = '''"""Toy module.

More about it.
"""

LIMIT = 3


def scale(x: float, factor: float = 2.0, *, clip: bool = False) -> float:
    """Multiply *x*.

    Args:
        x: The value.
        factor: The multiplier.
    """
    def inner(y):
        return y
    return x * factor


def _helper(a, *rest, **opts):
    return a


class Box:
    """A box."""

    size: int

    def area(self) -> int:
        """Its area."""
        return self.size ** 2
'''


def test_module_docstring_lines_and_constants():
    rec = module_record("src/toy.py", SOURCE)
    assert rec.path == "src/toy.py"
    assert rec.doc.startswith("Toy module.")
    assert rec.constants == ("LIMIT",)
    assert rec.n_lines == len(SOURCE.splitlines())


def test_signature_keeps_annotations_defaults_and_keyword_only_marker():
    scale = module_record("src/toy.py", SOURCE).defs[0]
    assert scale.signature == (
        "scale(x: float, factor: float = 2.0, *, clip: bool = False) -> float"
    )
    assert scale.returns_annotation == "float"
    assert dict(scale.params) == {"x": "float", "factor": "float", "clip": "bool"}


def test_varargs_and_kwargs_appear_in_the_signature():
    helper = module_record("src/toy.py", SOURCE).defs[1]
    assert helper.signature == "_helper(a, *rest, **opts)"
    assert helper.doc.summary == ""


def test_nested_functions_and_methods_are_children_with_qualified_names():
    defs = module_record("src/toy.py", SOURCE).defs
    scale, box = defs[0], defs[2]
    assert [(c.kind, c.qualname) for c in scale.children] == [("nested", "scale.inner")]
    assert box.kind == "class" and box.fields == ("size: int",)
    assert [(c.kind, c.qualname) for c in box.children] == [("method", "Box.area")]
    assert "self" not in dict(box.children[0].params)


def test_a_syntax_error_names_the_file():
    with pytest.raises(SyntaxError, match="src/broken.py"):
        module_record("src/broken.py", "def f(:\n")


NOTEBOOK = {
    "cells": [
        {"cell_type": "code", "source": ["import numpy as np\n"]},
        {"cell_type": "markdown", "source": ["# Title\n", "What it does."]},
        {"cell_type": "code", "source": ["data = load_trials(root)\n", "print(len(data))"]},
        {"cell_type": "markdown", "source": ["More words, no heading."]},
        {"cell_type": "markdown", "source": ["## 1. Fit\n", "Fit things."]},
        {"cell_type": "code", "source": ["fit = fit_model(data)\n", "ax.set_title('x')"]},
        {"cell_type": "code", "source": ["plot_fit(fit)"]},
    ]
}


def test_notebook_blocks_follow_the_headings_and_count_code_cells():
    rec = notebook_record("scripts/toy.ipynb", NOTEBOOK)
    assert (rec.n_cells, rec.n_code) == (7, 4)
    assert [(b.heading, b.level, b.n_code) for b in rec.blocks] == [
        ("(setup)", 1, 1),
        ("Title", 1, 1),
        ("1. Fit", 2, 2),
    ]


def test_notebook_blocks_join_heading_free_markdown_and_list_called_functions():
    blocks = notebook_record("scripts/toy.ipynb", NOTEBOOK).blocks
    assert blocks[1].text == "What it does. More words, no heading."
    assert blocks[1].calls == ("load_trials",)
    assert blocks[2].calls == ("fit_model", "plot_fit")
