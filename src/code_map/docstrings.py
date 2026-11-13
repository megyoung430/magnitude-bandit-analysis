"""Parse Google-style docstrings into the parts the code map shows (pure)."""

import re
import textwrap
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
# "- x", "* x", "1. x": an explicit list item
BULLET = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+")
# "'key' - text", "name: text", "name (type): text": a definition-style line
KEYED = re.compile(r"^(?:['\"][^'\"]+['\"]|`?\w[\w.]*`?)(?:\s*\([^)]*\))?\s*(?:-|:)\s")
SENTENCE_END = re.compile(r"(?<=[a-z0-9)\]'\"`][.!?])\s+(?=[A-Z`'\"(])")
RETURNS_SENTENCE = re.compile(r"(?:^|(?<=[.!?])\s+)(?=(?:Returns?|Yields?)\b)")


class Arg(NamedTuple):
    """One documented argument."""

    name: str
    type: str
    description: str


class DocParts(NamedTuple):
    """The parts of a docstring.

    Attributes:
        summary: The first paragraph, on one line.
        body: The rest of the text before the first section, as light Markdown: paragraphs
            separated by blank lines, list items as "- " lines indented by depth.
        args: Documented arguments, in order.
        returns: The Returns (or Yields) section, or else a "Returns ..." sentence of the body
            with the list that follows it, as light Markdown.
        raises: The Raises section, one "- " item per exception when there are several.
    """

    summary: str
    body: str
    args: tuple[Arg, ...]
    returns: str
    raises: str


def one_line(text: str) -> str:
    """Collapse all whitespace in *text* to single spaces."""
    return " ".join(text.split())


class Entry(NamedTuple):
    """One paragraph (depth -1) or list item (depth 0, 1, ...) of structured text."""

    depth: int
    text: str


def entry_text(entry: Entry) -> str:
    """An entry as one line of the light Markdown the page renders."""
    return entry.text if entry.depth < 0 else "  " * entry.depth + "- " + entry.text


def render(blocks: list[list[Entry]]) -> str:
    """Blocks as text: entries on their own lines, blocks separated by a blank line."""
    return "\n\n".join("\n".join(entry_text(e) for e in b) for b in blocks if b)


def structure_blocks(lines: list[str]) -> list[list[Entry]]:
    """Paragraphs and (nested) list items of some docstring lines.

    Wrapped lines join their paragraph or item. A deeper-indented line starts a list item when
    the line before ends with a colon, when it looks like a definition ("'key' - text",
    "name: text") or when it lines up with the item before; otherwise it continues that line.
    Explicit bullets ("- ", "* ", "1. ") always start an item.
    """
    blocks: list[list[Entry]] = [[]]
    stack: list[int] = []  # indents of the open list levels
    for raw in textwrap.dedent("\n".join(lines)).splitlines():
        text, indent = raw.strip(), len(raw) - len(raw.lstrip())
        if not text:
            if blocks[-1]:
                blocks.append([])
                stack = []
            continue
        last = blocks[-1][-1] if blocks[-1] else None
        bullet = BULLET.match(text)
        if bullet:
            text = text[bullet.end() :]
        if last is None:
            starts_item = bool(bullet) or indent > 0 and bool(KEYED.match(text))
        elif last.depth < 0:
            starts_item = (
                bool(bullet) or indent > 0 and (last.text.endswith(":") or bool(KEYED.match(text)))
            )
        else:
            item_indent = stack[-1] if stack else 0
            starts_item = (
                bool(bullet)
                or indent <= item_indent
                or (last.text.endswith(":") or bool(KEYED.match(text)))
            )
            if indent == 0 and not bullet and item_indent > 0:
                starts_item, last = False, None  # back to the margin: a new paragraph
        if starts_item:
            while stack and stack[-1] > indent:
                stack.pop()
            if not stack or stack[-1] < indent:
                stack.append(indent)
            blocks[-1].append(Entry(len(stack) - 1, text))
        elif last is not None:
            blocks[-1][-1] = Entry(last.depth, f"{last.text} {text}")
        else:
            stack = []
            blocks[-1].append(Entry(-1, text))
    return [[Entry(e.depth, one_line(e.text)) for e in b] for b in blocks if b]


def structure(lines: list[str]) -> str:
    """Docstring lines as light Markdown: paragraphs, and "- " items indented by depth."""
    return render(structure_blocks(lines))


def split_returns(blocks: list[list[Entry]]) -> tuple[list[list[Entry]], list[list[Entry]]]:
    """Move the first "Returns ..." (or "Yields ...") sentence of a paragraph, the rest of that
    paragraph and the list right after it out of *blocks*; returns (remaining, moved).

    This finds the output of docstrings written without a Returns section.
    """
    for b, block in enumerate(blocks):
        for i, entry in enumerate(block):
            if entry.depth >= 0:
                continue
            match = RETURNS_SENTENCE.search(entry.text)
            if not match:
                continue
            before, moved = entry.text[: match.start()].strip(), entry.text[match.start() :].strip()
            end = i + 1
            while end < len(block) and block[end].depth >= 0:
                end += 1
            kept = block[:i] + ([Entry(-1, before)] if before else []) + block[end:]
            out = [Entry(-1, moved), *block[i + 1 : end]]
            return blocks[:b] + ([kept] if kept else []) + blocks[b + 1 :], [out]
    return blocks, []


def split_summary(blocks: list[list[Entry]], first_lines: int) -> tuple[str, list[list[Entry]]]:
    """The summary and the remaining blocks.

    A one-line first paragraph is the whole summary (the Google convention). When the first
    paragraph spans several lines, only its first sentence is, and the rest opens the body.
    """
    if not blocks or blocks[0][0].depth >= 0:
        return "", blocks
    first, rest = blocks[0][0].text, blocks[0][1:]
    if first_lines > 1:
        head, *tail = SENTENCE_END.split(first, maxsplit=1)
        if tail:
            return head, [[Entry(-1, tail[0]), *rest], *blocks[1:]]
    return first, ([rest] if rest else []) + blocks[1:]


def parse_raises(lines: list[str]) -> str:
    """The Raises section, one item per exception when there are several."""
    entries: list[str] = []
    base = None
    for line in textwrap.dedent("\n".join(lines)).splitlines():
        indent = len(line) - len(line.lstrip())
        if not line.strip():
            continue
        if base is None or indent <= base:
            base = indent
            entries.append(line.strip())
        else:
            entries[-1] += " " + line.strip()
    entries = [one_line(e) for e in entries]
    return entries[0] if len(entries) == 1 else "\n".join(f"- {e}" for e in entries)


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
    body_lines = sections["body"]
    first_lines = next(
        (i for i, line in enumerate(body_lines) if not line.strip()), len(body_lines)
    )
    summary, blocks = split_summary(structure_blocks(body_lines), first_lines)
    args = tuple(a for name in ARG_SECTIONS for a in parse_args(sections.get(name, [])))
    returns = structure([line for n in RETURN_SECTIONS for line in sections.get(n, [])])
    if not returns:
        blocks, moved = split_returns(blocks)
        returns = render(moved)
    raises = parse_raises(sections.get("Raises", []))
    return DocParts(summary, render(blocks), args, returns, raises)
