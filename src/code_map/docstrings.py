"""Parse Google-style docstrings into the parts the code map shows (pure)."""

import re
from typing import NamedTuple

SECTION_NAMES = (
    "Args", "Arguments", "Parameters", "Returns", "Return", "Yields", "Raises",
    "Attributes", "Notes", "Note", "Examples", "Example", "References",
)  # fmt: skip
ARG_SECTIONS = ("Args", "Arguments", "Parameters")
RETURN_SECTIONS = ("Returns", "Return", "Yields")
SECTION_HEADER = re.compile(r"^\s*(%s):\s*$" % "|".join(SECTION_NAMES))
# "name (type): text" or "name: text", indented at most 8 spaces under the section header
ARG_LINE = re.compile(r"^(\s{0,8})(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


class Arg(NamedTuple):
    """One documented argument."""

    name: str
    type: str
    description: str


class DocParts(NamedTuple):
    """The parts of a docstring.

    Attributes:
        summary: The first paragraph, on one line.
        body: The other paragraphs before the first section, on one line.
        args: Documented arguments, in order.
        returns: The Returns (or Yields) section, on one line.
        raises: The Raises section, on one line.
    """

    summary: str
    body: str
    args: tuple[Arg, ...]
    returns: str
    raises: str


def one_line(text: str) -> str:
    """Collapse all whitespace in *text* to single spaces."""
    return " ".join(text.split())


def split_sections(doc: str) -> dict[str, list[str]]:
    """Lines of *doc* grouped under their section header ("body" for those before any header)."""
    sections: dict[str, list[str]] = {"body": []}
    current = "body"
    for line in doc.splitlines():
        header = SECTION_HEADER.match(line)
        if header:
            current = header.group(1)
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return sections


def parse_args(lines: list[str]) -> tuple[Arg, ...]:
    """The arguments of an Args section; deeper-indented lines continue the previous one."""
    args: list[list[str]] = []
    base = None
    for line in lines:
        match = ARG_LINE.match(line)
        indent = len(line) - len(line.lstrip())
        if match and (base is None or indent <= base):
            base = indent
            args.append([match.group(2).lstrip("*"), match.group(3) or "", match.group(4)])
        elif args and line.strip():
            args[-1][2] += " " + line.strip()
    return tuple(Arg(name, kind, one_line(text)) for name, kind, text in args)


def parse_docstring(doc: str | None) -> DocParts:
    """Split a Google-style docstring into summary, body, arguments, returns and raises.

    Args:
        doc: The docstring (already dedented, as ``ast.get_docstring`` returns it), or None.

    Returns:
        DocParts with empty strings and an empty tuple for whatever is missing.
    """
    if not doc:
        return DocParts("", "", (), "", "")
    sections = split_sections(doc)
    paragraphs = [p for p in re.split(r"\n\s*\n", "\n".join(sections["body"]).strip()) if p]
    summary = one_line(paragraphs[0]) if paragraphs else ""
    body = one_line(" ".join(paragraphs[1:]))
    args = tuple(a for name in ARG_SECTIONS for a in parse_args(sections.get(name, [])))
    returns = one_line(" ".join(" ".join(sections.get(n, [])) for n in RETURN_SECTIONS))
    raises = one_line(" ".join(sections.get("Raises", [])))
    return DocParts(summary, body, args, returns, raises)
