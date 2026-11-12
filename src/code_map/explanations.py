"""Parse ``docs/code_map/explanations.md``: plain-language explanations for readers new to the
methods, keyed by file path (a primer for the whole file) or ``path::qualname`` (one function).

The file is Markdown so it reads well on GitHub: each entry starts with a ``## <key>`` heading
and runs to the next one. Paragraphs, ``- `` bullet lists, backticks and ``**bold**`` render on
the page. Text before the first entry is a preamble for people editing the file.
"""

import re

HEADING = re.compile(r"^## +(.+?)\s*$")
KEY = re.compile(r"^(src|scripts)/[\w./-]+\.(py|ipynb)(::[\w.]+)?$")


def parse_explanations(text: str) -> dict[str, str]:
    """``key → explanation`` for every ``## key`` entry, in file order.

    Raises:
        ValueError: For a heading that is not a file path (or ``path::qualname``), a key given
            twice, or an entry with no text; the message gives the line number.
    """
    entries: dict[str, list[str]] = {}
    starts: dict[str, int] = {}
    current = None
    for number, line in enumerate(text.splitlines(), start=1):
        heading = HEADING.match(line)
        if heading:
            key = heading.group(1).strip("`")
            if not KEY.match(key):
                raise ValueError(f"explanations line {number}: {key!r} is not a path or path::name")
            if key in entries:
                raise ValueError(f"explanations line {number}: {key} is explained twice")
            entries[key], starts[key], current = [], number, key
        elif current is not None:
            entries[current].append(line)
    found = {}
    for key, lines in entries.items():
        body = "\n".join(lines).strip()
        if not body:
            raise ValueError(f"explanations line {starts[key]}: the entry for {key} is empty")
        found[key] = re.sub(r"\n{3,}", "\n\n", body)
    return found
