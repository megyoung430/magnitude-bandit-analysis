"""Notion page content and row properties for the code map (pure: builds Notion API JSON).

One database row per file. A row's page holds the file's summary and one toggle per function or
class, whose children give the description, inputs, output, errors and nested definitions. The
Notion API accepts at most 2000 characters per rich-text item, 100 blocks per request and two
levels of nesting per request, so text is chunked, toggles hold only flat children, and
:func:`request_batches` splits a page's blocks across requests.
"""

import hashlib
import html
import json
import re

# Bump when the page layout changes, so the next sync rewrites every page.
LAYOUT_VERSION = 1
MAX_TEXT = 2000
MAX_BLOCKS_PER_REQUEST = 100
MAX_NESTED_PER_REQUEST = 900  # the API limit is 1000 blocks in total, children included
KIND_LABELS = {"f": "function", "c": "class", "m": "method", "n": "inner function"}
CODE_SPAN = re.compile(r"``([^`]+)``|`([^`]+)`")
TAG = re.compile(r"<[^>]+>")


def plain_text(text: str) -> str:
    """Overview HTML as plain text: ``<code>x</code>`` becomes ``x`` in backticks, other tags go."""
    text = re.sub(r"<code>(.*?)</code>", r"``\1``", text)
    return html.unescape(TAG.sub("", text))


def text_item(content: str, **annotations: bool | str) -> dict:
    """One rich-text item (at most 2000 characters) with optional annotations, e.g. code=True
    or color="gray"."""
    item = {"type": "text", "text": {"content": content}}
    if annotations:
        item["annotations"] = annotations
    return item


def chunks(text: str, size: int = MAX_TEXT) -> list[str]:
    """*text* cut into pieces of at most *size* characters."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def rich_text(text: str, **annotations: bool | str) -> list[dict]:
    """Rich-text items for *text*, with ``code`` spans marked as code and long runs chunked."""
    items, last = [], 0
    for match in CODE_SPAN.finditer(text):
        if match.start() > last:
            items += [text_item(c, **annotations) for c in chunks(text[last : match.start()])]
        code = match.group(1) or match.group(2)
        items += [text_item(c, **annotations, code=True) for c in chunks(code)]
        last = match.end()
    if last < len(text) or not items:
        items += [text_item(c, **annotations) for c in chunks(text[last:])]
    return items


def block(kind: str, items: list[dict], children: list[dict] | None = None, **extra) -> dict:
    """A Notion block of *kind* (paragraph, toggle, ...) holding *items* as its rich text."""
    body = {"rich_text": items, **extra}
    if children:
        body["children"] = children
    return {"object": "block", "type": kind, kind: body}


def paragraph(text: str, **annotations: bool | str) -> dict:
    """A paragraph block."""
    return block("paragraph", rich_text(text, **annotations))


def labeled(label: str, text: str) -> dict:
    """A paragraph starting with a bold label, e.g. "Output: dict: The fit."."""
    return block("paragraph", [text_item(f"{label}: ", bold=True), *rich_text(text)])


def bullet(items: list[dict]) -> dict:
    """A bulleted list item."""
    return block("bulleted_list_item", items)


def input_bullet(name: str, kind: str, description: str) -> dict:
    """One input as "name (type): description"."""
    items = [text_item(name, code=True)]
    items += [text_item(f" ({kind})", italic=True)] if kind else []
    items += [text_item(": "), *rich_text(description)] if description else []
    return bullet(items)


def child_bullet(entry: dict) -> dict:
    """A nested definition (method or inner function) as one bullet: signature, then description."""
    items = [text_item(entry["s"], code=True)]
    if entry.get("d"):
        items += [text_item(" — "), *rich_text(entry["d"])]
    if entry.get("o"):
        items += [text_item(" Output: ", bold=True), *rich_text(entry["o"])]
    return bullet(items)


def definition_children(entry: dict) -> list[dict]:
    """The flat blocks inside a definition's toggle."""
    description = entry.get("d") or "No description."
    blocks = [block("paragraph", rich_text(description))]
    if entry.get("u"):
        note = " (written from the code; no docstring)"
        blocks[0]["paragraph"]["rich_text"].append(text_item(note, italic=True, color="gray"))
    if entry.get("b"):
        blocks.append(block("paragraph", rich_text(entry["b"]), color="gray"))
    if entry.get("i"):
        blocks.append(labeled("Inputs", ""))
        blocks += [input_bullet(*row) for row in entry["i"]]
    if entry.get("f"):
        blocks.append(labeled("Fields", ""))
        blocks += [bullet([text_item(field, code=True)]) for field in entry["f"]]
    if entry.get("o"):
        blocks.append(labeled("Output", entry["o"]))
    if entry.get("r"):
        blocks.append(labeled("Raises", entry["r"]))
    if entry.get("c"):
        title = "Methods" if entry["k"] == "c" else "Inner functions"
        blocks.append(labeled(title, ""))
        blocks += [child_bullet(c) for c in entry["c"]]
    return blocks


def definition_block(entry: dict) -> dict:
    """A toggle titled with the definition's kind and signature."""
    title = [text_item(KIND_LABELS.get(entry["k"], "") + "  ", color="gray")]
    title += [text_item(c, code=True) for c in chunks(entry["s"])]
    return block("toggle", title, definition_children(entry))


def header_blocks(summary: str, more: str, meta: str) -> list[dict]:
    """The top of a file's page: summary callout, longer description and provenance."""
    blocks = [block("callout", rich_text(plain_text(summary) or "No description."),
                    icon={"type": "emoji", "emoji": "📄"}, color="gray_background")]  # fmt: skip
    if more:
        blocks.append(paragraph(plain_text(more)))
    blocks.append(block("paragraph", rich_text(f"Synced from {meta}."), color="gray"))
    return blocks


def module_blocks(mod: dict, meta: str) -> list[dict]:
    """The page content of one Python file."""
    blocks = header_blocks(mod["a"], mod.get("x", ""), meta)
    if mod.get("K"):
        blocks.append(labeled("Constants", ", ".join(f"``{k}``" for k in mod["K"])))
    if not mod["fn"]:
        blocks.append(paragraph("No functions: the file runs top to bottom, or only re-exports."))
        return blocks
    blocks.append(block("heading_3", rich_text("Functions and classes")))
    return blocks + [definition_block(e) for e in mod["fn"]]


def notebook_blocks(mod: dict, meta: str) -> list[dict]:
    """The page content of one notebook: its heading blocks, their text and the calls they make."""
    blocks = header_blocks(mod["a"], "", meta)
    blocks.append(paragraph(f"{mod['code']} code cells of {mod['cells']}."))
    for b in mod["blocks"]:
        heading = f"{b['h']}  ({b['c']} code cell{'' if b['c'] == 1 else 's'})"
        kind = "heading_3" if b["v"] <= 2 else "paragraph"
        blocks.append(block(kind, rich_text(heading, bold=kind == "paragraph")))
        if b["t"]:
            blocks.append(paragraph(b["t"]))
        if b["f"]:
            blocks.append(labeled("Calls", ", ".join(f"``{c}``" for c in b["f"])))
    return blocks


def page_blocks(mod: dict, meta: str) -> list[dict]:
    """The page content of one file or notebook."""
    return notebook_blocks(mod, meta) if mod.get("nb") else module_blocks(mod, meta)


def count_definitions(entries: list[dict]) -> int:
    """Definitions in *entries*, nested ones included."""
    return sum(1 + count_definitions(e.get("c", [])) for e in entries)


def content_hash(mod: dict, folder: str) -> str:
    """Fingerprint of everything a file's row shows, plus the layout version.

    The commit a page was synced from is left out on purpose: a file whose content is unchanged
    keeps its page, even though the "Synced from" line then names an older commit.
    """
    payload = json.dumps([LAYOUT_VERSION, folder, mod], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def row_properties(mod: dict, folder: str, meta: str, digest: str) -> dict:
    """The database properties of one file's row.

    Args:
        mod: The file's entry from :func:`src.code_map.assemble.build_map`.
        folder: The folder it is listed under.
        meta: Where the sync ran from ("branch @ commit, date").
        digest: Its :func:`content_hash` ("" while the page is still being written).
    """
    n_defs = 0 if mod.get("nb") else count_definitions(mod["fn"])
    return {
        "File": {"title": [text_item(mod["p"])]},
        "Folder": {"select": {"name": folder}},
        "Type": {"select": {"name": "Notebook" if mod.get("nb") else "Python"}},
        "Summary": {"rich_text": [text_item(chunks(plain_text(mod["a"]))[0])]},
        "Lines": {"number": mod.get("L", mod.get("cells"))},
        "Definitions": {"number": n_defs},
        "Built From": {"rich_text": [text_item(meta)]},
        "Source Hash": {"rich_text": [text_item(digest)] if digest else []},
    }


def block_size(b: dict) -> int:
    """Blocks a request counts for *b*: itself and its children."""
    return 1 + len(b[b["type"]].get("children", []))


def request_batches(blocks: list[dict]) -> list[list[dict]]:
    """*blocks* split into consecutive batches that each fit one Notion request."""
    batches: list[list[dict]] = [[]]
    size = 0
    for b in blocks:
        n = block_size(b)
        full = len(batches[-1]) == MAX_BLOCKS_PER_REQUEST or size + n > MAX_NESTED_PER_REQUEST
        if batches[-1] and full:
            batches.append([])
            size = 0
        batches[-1].append(b)
        size += n
    return batches if batches[0] else []
