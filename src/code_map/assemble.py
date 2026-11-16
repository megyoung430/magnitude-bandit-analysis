"""The code-map data: extracted records joined with written descriptions and overviews (pure).

Descriptions come from docstrings. A definition with no docstring takes its description from
``docs/code_map/descriptions.tsv`` (keyed ``path::qualname``) and is flagged, so the page can
say the text was written from the code rather than by its author.
"""

from collections.abc import Iterable, Mapping
from typing import NamedTuple

from src.code_map.docstrings import one_line, structure
from src.code_map.extract import Definition, ModuleRecord, NotebookRecord

MAX_CONSTANTS = 30
MAX_BLOCK_TEXT = 420
MAX_BLOCK_CALLS = 14
SIDE_EFFECT = "None (acts by side effect)"


class Overviews(NamedTuple):
    """Hand-written text for the page (``docs/code_map/overviews.toml``).

    Attributes:
        intro: The paragraph under the page title (HTML allowed).
        sections: (name, text) for each part of the repo, in display order.
        packages: Folder name → summary; the order here is the page's folder order.
        modules: File path → summary, for files whose docstring is missing or too terse.
        notebooks: Notebook path → summary.
    """

    intro: str
    sections: tuple[tuple[str, str], ...]
    packages: dict[str, str]
    modules: dict[str, str]
    notebooks: dict[str, str]


def overviews_from_toml(parsed: Mapping) -> Overviews:
    """Overviews from the parsed TOML file.

    Raises:
        ValueError: If the file has no ``intro``.
    """
    if "intro" not in parsed:
        raise ValueError("overviews.toml needs an 'intro' string")
    return Overviews(
        parsed["intro"],
        tuple((s["name"], s["text"]) for s in parsed.get("section", [])),
        dict(parsed.get("package", {})),
        dict(parsed.get("module", {})),
        dict(parsed.get("notebook", {})),
    )


def parse_descriptions(text: str) -> dict[str, str]:
    """``path::qualname → description`` from the TSV text; blank lines and ``#`` lines are skipped.

    Raises:
        ValueError: For a line without a tab, a key without ``::``, or a key given twice.
    """
    found: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        key, tab, description = line.partition("\t")
        if not tab or "::" not in key or not description.strip():
            raise ValueError(
                f"descriptions line {number} is not 'path::qualname<TAB>text': {line!r}"
            )
        if key in found:
            raise ValueError(f"descriptions line {number}: {key} is described twice")
        found[key] = description.strip()
    return found


def clip(text: str, limit: int) -> str:
    """*text* cut to at most *limit* characters, with an ellipsis if cut.

    Multi-line text (paragraphs and list items) is cut at a line break so a list item is never
    split; single-line text at a word boundary.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rsplit("\n", 1)[0] if "\n" in head else head.rsplit(" ", 1)[0]
    return cut.rstrip() + " …"


def inputs(d: Definition) -> list[list[str]]:
    """[name, type, description] per input: signature parameters, then documented-only ones.

    The type is the annotation, or the docstring's type when there is no annotation.
    """
    documented = {a.name: a for a in d.doc.args}
    rows = []
    for name, annotation in d.params:
        arg = documented.get(name)
        rows.append([name, annotation or (arg.type if arg else ""), arg.description if arg else ""])
    named = {name for name, _ in d.params}
    rows += [[a.name, a.type, a.description] for a in d.doc.args if a.name not in named]
    return rows


def output(d: Definition) -> str:
    """The documented return value, else the return annotation ("None" means a side effect)."""
    if d.doc.returns:
        return d.doc.returns
    return SIDE_EFFECT if d.returns_annotation == "None" else d.returns_annotation


def definition_entry(d: Definition, path: str, descriptions: Mapping[str, str]) -> dict:
    """The page entry of one definition and, recursively, the definitions nested in it.

    Keys are short because the page embeds ~3000 of them: n name, k kind letter, s signature,
    d description, u written-from-code flag, l line, b body, f fields, i inputs, o output,
    r raises, c children.
    """
    written = descriptions.get(f"{path}::{d.qualname}", "")
    entry = {"n": d.qualname, "k": d.kind[0], "s": d.signature, "d": d.doc.summary or written}
    entry["l"] = d.line
    if not d.doc.summary and written:
        entry["u"] = 1
    if d.doc.body:
        entry["b"] = d.doc.body
    if d.kind == "class":
        if d.fields:
            entry["f"] = list(d.fields)
    else:
        if rows := inputs(d):
            entry["i"] = rows
        if out := output(d):
            entry["o"] = out
        if d.doc.raises:
            entry["r"] = d.doc.raises
    if d.children:
        entry["c"] = [definition_entry(c, path, descriptions) for c in d.children]
    return entry


def walk(defs: Iterable[Definition]) -> Iterable[Definition]:
    """Every definition, depth first."""
    for d in defs:
        yield d
        yield from walk(d.children)


def coverage(
    modules: Iterable[ModuleRecord], descriptions: Mapping[str, str]
) -> tuple[list[str], list[str]]:
    """Keys of definitions with neither docstring nor description, and descriptions with no
    definition (renamed, removed, or on a branch that is not checked out).

    Nested functions are not required to have a description; they show under their parent.
    """
    keys, missing = set(), []
    for m in modules:
        for d in walk(m.defs):
            key = f"{m.path}::{d.qualname}"
            keys.add(key)
            if d.kind != "nested" and not d.doc.summary and key not in descriptions:
                missing.append(key)
    return missing, sorted(set(descriptions) - keys)


def package_of(path: str) -> str:
    """The folder a file is listed under: its first two path parts, or "<top> (top level)"."""
    parts = path.split("/")
    return f"{parts[0]} (top level)" if len(parts) == 2 else "/".join(parts[:2])


def module_entry(m: ModuleRecord, overviews: Overviews, descriptions: Mapping[str, str]) -> dict:
    """The page entry of one Python file."""
    first, _, rest = m.doc.strip().partition("\n\n")
    entry = {
        "p": m.path,
        "a": overviews.modules.get(m.path) or one_line(first),
        "x": structure(rest.splitlines()),
        "L": m.n_lines,
        "fn": [definition_entry(d, m.path, descriptions) for d in m.defs],
    }
    if m.constants:
        entry["K"] = list(m.constants[:MAX_CONSTANTS])
    return entry


def notebook_entry(nb: NotebookRecord, overviews: Overviews) -> dict:
    """The page entry of one notebook."""
    blocks = [
        {"h": b.heading, "v": b.level, "t": clip(b.text, MAX_BLOCK_TEXT), "c": b.n_code,
         "f": list(b.calls[:MAX_BLOCK_CALLS])}
        for b in nb.blocks
    ]  # fmt: skip
    return {"p": nb.path, "nb": 1, "a": overviews.notebooks.get(nb.path, ""), "cells": nb.n_cells,
            "code": nb.n_code, "blocks": blocks}  # fmt: skip


def build_map(
    modules: Iterable[ModuleRecord],
    notebooks: Iterable[NotebookRecord],
    overviews: Overviews,
    descriptions: Mapping[str, str],
    meta: str,
    explanations: Mapping[str, str] | None = None,
) -> dict:
    """Everything the page shows, as JSON-ready data.

    Args:
        modules: Every Python file.
        notebooks: Every notebook.
        overviews: The hand-written page text.
        descriptions: Written descriptions for definitions without a docstring.
        meta: Where the map was built from, e.g. "main @ 9868047, 2026-11-12".
        explanations: Plain-language explanations keyed by path or ``path::qualname``.

    Returns:
        A dict with intro, sections, meta and pkgs (folders in overview order, then any folder
        the overviews do not list); notebooks come before scripts within a scripts folder.
    """
    pkgs = {name: {"name": name, "about": about, "mods": []}
            for name, about in overviews.packages.items()}  # fmt: skip
    for m in modules:
        name = package_of(m.path)
        pkgs.setdefault(name, {"name": name, "about": "", "mods": []})
        pkgs[name]["mods"].append(module_entry(m, overviews, descriptions))
    for nb in notebooks:
        name = package_of(nb.path)
        pkgs.setdefault(name, {"name": name, "about": "", "mods": []})
        pkgs[name]["mods"].append(notebook_entry(nb, overviews))
    for p in pkgs.values():
        p["mods"].sort(key=lambda e: (not e.get("nb"), e["p"]))
    attach_explanations(pkgs.values(), explanations or {})
    return {
        "intro": overviews.intro,
        "sections": [list(s) for s in overviews.sections],
        "meta": meta,
        "pkgs": [p for p in pkgs.values() if p["mods"]],
    }


def attach_explanation(entries: list[dict], path: str, explanations: Mapping[str, str]) -> None:
    """Put each definition's plain-language explanation (key ``path::qualname``) in its "w"."""
    for e in entries:
        if text := explanations.get(f"{path}::{e['n']}"):
            e["w"] = text
        attach_explanation(e.get("c", []), path, explanations)


def attach_explanations(pkgs: Iterable[dict], explanations: Mapping[str, str]) -> None:
    """Add the explanations to the page data in place: a file's primer and its functions'."""
    for pkg in pkgs:
        for mod in pkg["mods"]:
            if text := explanations.get(mod["p"]):
                mod["w"] = text
            attach_explanation(mod.get("fn", []), mod["p"], explanations)


def stale_explanations(
    modules: Iterable[ModuleRecord], notebooks: Iterable[NotebookRecord], keys: Iterable[str]
) -> list[str]:
    """Explanation keys that name no file, notebook or definition in this checkout."""
    known = {nb.path for nb in notebooks}
    for m in modules:
        known.add(m.path)
        known.update(f"{m.path}::{d.qualname}" for d in walk(m.defs))
    return sorted(set(keys) - known)
