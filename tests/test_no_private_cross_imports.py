"""A module must not import another module's underscore-prefixed names.

A leading underscore says "internal to this module"; importing it elsewhere ties two modules
together invisibly, and renaming it for tidiness breaks the other one. Shared helpers are public.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def private_imports(tree: ast.AST) -> list[tuple[int, str, str]]:
    """``(line, module, name)`` for each ``from src... import _name`` in *tree*."""
    return [
        (node.lineno, node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src")
        for alias in node.names
        if alias.name.startswith("_") and not alias.name.startswith("__")
    ]


def test_private_imports_are_found_by_the_scan():
    tree = ast.parse("from src.a import _hidden, shown\nfrom os import _exit\n")
    assert private_imports(tree) == [(1, "src.a", "_hidden")]


def test_no_module_imports_a_private_name_from_another_src_module():
    found = {
        str(path.relative_to(SRC)): hits
        for path in sorted(SRC.rglob("*.py"))
        if (hits := private_imports(ast.parse(path.read_text())))
    }
    assert found == {}
