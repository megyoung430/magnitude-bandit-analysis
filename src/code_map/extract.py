"""Definitions of a Python file and blocks of a notebook, read from their text (pure)."""

import ast
import re
from typing import NamedTuple

from src.code_map.docstrings import DocParts, one_line, parse_docstring

# Calls too generic to say anything about what a notebook block does.
IGNORED_CALLS = frozenset(
    "print range len list dict sorted zip enumerate display float isinstance tuple str int set "
    "super getattr max min sum round open format items keys values append join exists mkdir copy "
    "update get insert".split()
)
CALL = re.compile(r"\b([a-z_][a-z0-9_]{3,})\(")
MAX_BLOCK_TEXT = 900


class Definition(NamedTuple):
    """A function, method or class, with the definitions nested inside it.

    Attributes:
        kind: "function", "class", "method" or "nested".
        qualname: Dotted name inside the module, e.g. "Box.area".
        signature: The signature as written, e.g. "f(x: int = 1) -> str".
        params: (name, annotation) per parameter, without self/cls.
        returns_annotation: The return annotation ("" if none).
        doc: The parsed docstring.
        fields: Annotated class attributes, e.g. "size: int" (classes only).
        line: First line in the file.
        children: Methods and nested functions, in order.
    """

    kind: str
    qualname: str
    signature: str
    params: tuple[tuple[str, str], ...]
    returns_annotation: str
    doc: DocParts
    fields: tuple[str, ...]
    line: int
    children: tuple["Definition", ...]


class ModuleRecord(NamedTuple):
    """One Python file."""

    path: str
    doc: str
    n_lines: int
    constants: tuple[str, ...]
    defs: tuple[Definition, ...]


class NotebookBlock(NamedTuple):
    """The cells under one markdown heading."""

    heading: str
    level: int
    text: str
    n_code: int
    calls: tuple[str, ...]


class NotebookRecord(NamedTuple):
    """One notebook."""

    path: str
    n_cells: int
    n_code: int
    blocks: tuple[NotebookBlock, ...]


def annotation(node: ast.expr | None) -> str:
    """Source text of an annotation or default ("" for None)."""
    return ast.unparse(node) if node is not None else ""


def parameter_text(arg: ast.arg, default: ast.expr | None) -> str:
    """One parameter as written: name, annotation and default."""
    text = arg.arg + (f": {annotation(arg.annotation)}" if arg.annotation is not None else "")
    return text + (f" = {annotation(default)}" if default is not None else "")


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The function's signature as written, without decorators or body."""
    a = node.args
    positional = a.posonlyargs + a.args
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    parts = [parameter_text(arg, d) for arg, d in zip(positional, defaults, strict=True)]
    if a.vararg:
        parts.append("*" + parameter_text(a.vararg, None))
    elif a.kwonlyargs:
        parts.append("*")
    parts += [parameter_text(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True)]
    if a.kwarg:
        parts.append("**" + parameter_text(a.kwarg, None))
    returns = f" -> {annotation(node.returns)}" if node.returns is not None else ""
    return f"{node.name}({', '.join(parts)}){returns}"


def parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[tuple[str, str], ...]:
    """(name, annotation) of every parameter except self and cls."""
    a = node.args
    every = a.posonlyargs + a.args + ([a.vararg] if a.vararg else []) + a.kwonlyargs
    every += [a.kwarg] if a.kwarg else []
    return tuple((p.arg, annotation(p.annotation)) for p in every if p.arg not in ("self", "cls"))


def definitions(body: list[ast.stmt], prefix: str, parent: str) -> tuple[Definition, ...]:
    """The functions and classes directly in *body*, each with its own nested definitions.

    Args:
        body: Statements of a module, class or function.
        prefix: Qualified name of the enclosing definition ("" at module level).
        parent: "module", "class" or "function": decides whether a def is a function, a
            method or a nested function.
    """
    found = []
    for node in body:
        qualname = f"{prefix}.{node.name}" if prefix and hasattr(node, "name") else ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = qualname or node.name
            kind = {"module": "function", "class": "method"}.get(parent, "nested")
            found.append(Definition(
                kind, qualname, signature(node), parameters(node), annotation(node.returns),
                parse_docstring(ast.get_docstring(node)), (), node.lineno,
                definitions(node.body, qualname, "function"),
            ))  # fmt: skip
        elif isinstance(node, ast.ClassDef):
            qualname = qualname or node.name
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            fields = tuple(
                f"{ast.unparse(s.target)}: {ast.unparse(s.annotation)}"
                for s in node.body
                if isinstance(s, ast.AnnAssign)
            )
            found.append(Definition(
                "class", qualname, f"class {node.name}({bases})", (), "",
                parse_docstring(ast.get_docstring(node)), fields, node.lineno,
                definitions(node.body, qualname, "class"),
            ))  # fmt: skip
    return tuple(found)


def constants(tree: ast.Module) -> tuple[str, ...]:
    """Names of the module-level UPPER_CASE assignments."""
    names = []
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(target, ast.Name):
            if target.id.isupper():
                names.append(target.id)
    return tuple(names)


def module_record(path: str, source: str) -> ModuleRecord:
    """Docstring, size, constants and definitions of one Python file.

    Args:
        path: The file's path relative to the repo root (used as its name).
        source: The file's text.

    Raises:
        SyntaxError: If the file does not parse; the message names the file.
    """
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as err:
        raise SyntaxError(f"{path} does not parse: {err}") from err
    return ModuleRecord(
        path, ast.get_docstring(tree) or "", len(source.splitlines()), constants(tree),
        definitions(tree.body, "", "module"),
    )  # fmt: skip


def cell_text(cell: dict) -> str:
    """A notebook cell's source as one string."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def called_functions(code: str) -> list[str]:
    """Functions called in *code*, in order of first call, without generic built-ins."""
    return [c for c in dict.fromkeys(CALL.findall(code)) if c not in IGNORED_CALLS]


def notebook_blocks(cells: list[dict]) -> tuple[NotebookBlock, ...]:
    """Group cells under their markdown headings; code before the first heading is "(setup)"."""
    blocks: list[dict] = []
    for cell in cells:
        text = cell_text(cell)
        if cell.get("cell_type") == "markdown":
            lines = text.splitlines()
            headings = [line for line in lines if line.lstrip().startswith("#")]
            prose = one_line(" ".join(line for line in lines if line not in headings))
            if headings:
                level = len(headings[0].lstrip()) - len(headings[0].lstrip().lstrip("#"))
                heading = headings[0].lstrip().lstrip("#").strip()
                blocks.append(dict(heading=heading, level=level, text=prose, n_code=0, calls=[]))
            elif blocks:
                blocks[-1]["text"] = one_line(f"{blocks[-1]['text']} {prose}")
            else:
                blocks.append(dict(heading="(preamble)", level=1, text=prose, n_code=0, calls=[]))
        elif cell.get("cell_type") == "code":
            if not blocks:
                blocks.append(dict(heading="(setup)", level=1, text="", n_code=0, calls=[]))
            blocks[-1]["n_code"] += 1
            new = [c for c in called_functions(text) if c not in blocks[-1]["calls"]]
            blocks[-1]["calls"] += new
    return tuple(
        NotebookBlock(
            b["heading"],
            b["level"],
            b["text"][:MAX_BLOCK_TEXT],
            b["n_code"],
            tuple(c for c in b["calls"] if not c.startswith("set_")),
        )  # fmt: skip
        for b in blocks
    )


def notebook_record(path: str, notebook: dict) -> NotebookRecord:
    """Cell counts and heading blocks of one notebook (its parsed JSON)."""
    cells = notebook.get("cells", [])
    n_code = sum(c.get("cell_type") == "code" for c in cells)
    return NotebookRecord(path, len(cells), n_code, notebook_blocks(cells))
