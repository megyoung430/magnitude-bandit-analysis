"""The Notion side of the code map: page blocks, the API client and the sync, with a fake API."""

import json

import pytest

from src.code_map.notion_api import NotionClient, NotionError
from src.code_map.notion_blocks import (
    MAX_BLOCKS_PER_REQUEST,
    MAX_TEXT,
    content_hash,
    definition_block,
    module_blocks,
    notebook_blocks,
    plain_text,
    request_batches,
    rich_text,
    row_properties,
)
from src.code_map.notion_sync import Row, plan_sync, sync

FUNCTION = {
    "n": "fit", "k": "f", "s": "fit(trials, n: int = 5) -> dict", "d": "Fit ``trials``.",
    "i": [["trials", "pd.DataFrame", "One row per trial."], ["n", "int", ""]],
    "o": "dict: The fit.", "r": "ValueError: If empty.",
}  # fmt: skip
CLASS = {
    "n": "Box", "k": "c", "s": "class Box()", "d": "", "u": 1, "f": ["size: int"],
    "c": [{"n": "Box.area", "k": "m", "s": "area(self) -> int", "d": "Its area.", "o": "int"}],
}  # fmt: skip
MODULE = {
    "p": "src/a.py",
    "a": "Toy <code>module</code>.",
    "x": "",
    "L": 40,
    "fn": [FUNCTION, CLASS],
}
NOTEBOOK = {
    "p": "scripts/nb.ipynb", "nb": 1, "a": "A notebook.", "cells": 3, "code": 2,
    "blocks": [{"h": "1. Fit", "v": 2, "t": "Fit things.", "c": 2, "f": ["fit_model"]}],
}  # fmt: skip


def contents(items):
    return "".join(i["text"]["content"] for i in items)


def test_rich_text_marks_code_spans_and_keeps_every_character():
    items = rich_text("Use ``fit`` on `x` now")
    assert contents(items) == "Use fit on x now"
    assert [i.get("annotations", {}).get("code", False) for i in items] == [
        False, True, False, True, False,
    ]  # fmt: skip


def test_rich_text_chunks_long_text_under_the_api_limit():
    items = rich_text("a" * (2 * MAX_TEXT + 5))
    assert [len(i["text"]["content"]) for i in items] == [MAX_TEXT, MAX_TEXT, 5]


def test_overview_html_becomes_plain_text_with_code_spans():
    assert plain_text("Reads <code>.tsv</code> files &amp; <b>more</b>.") == (
        "Reads ``.tsv`` files & more."
    )


def test_a_definition_is_a_toggle_whose_children_are_flat():
    toggle = definition_block(FUNCTION)
    assert toggle["type"] == "toggle"
    children = toggle["toggle"]["children"]
    assert all("children" not in c[c["type"]] for c in children)
    texts = [contents(c[c["type"]]["rich_text"]) for c in children]
    assert texts == [
        "Fit trials.", "Inputs: ", "trials (pd.DataFrame): One row per trial.", "n (int)",
        "Output: dict: The fit.", "Raises: ValueError: If empty.",
    ]  # fmt: skip


def test_a_class_lists_fields_and_methods_and_flags_a_written_description():
    children = definition_block(CLASS)["toggle"]["children"]
    texts = [contents(c[c["type"]]["rich_text"]) for c in children]
    assert texts[0].startswith("No description.")
    assert "Methods: " in texts and "area(self) -> int — Its area. Output: int" in texts
    assert "size: int" in texts


def test_module_and_notebook_pages_open_with_the_summary():
    page = module_blocks(MODULE, "main @ abc, 2026-11-12")
    assert page[0]["type"] == "callout"
    assert contents(page[0]["callout"]["rich_text"]) == "Toy module."
    assert [b["type"] for b in page[-2:]] == ["toggle", "toggle"]
    nb = notebook_blocks(NOTEBOOK, "main @ abc, 2026-11-12")
    assert [b["type"] for b in nb[-3:]] == ["heading_3", "paragraph", "paragraph"]


def test_batches_respect_the_block_and_nesting_limits():
    toggle = definition_block(FUNCTION)  # 1 block + 6 children
    batches = request_batches([toggle] * 250)
    assert all(len(b) <= MAX_BLOCKS_PER_REQUEST for b in batches)
    assert all(sum(1 + len(t["toggle"]["children"]) for t in b) <= 900 for b in batches)
    assert sum(len(b) for b in batches) == 250


def test_row_properties_name_the_file_folder_and_counts():
    props = row_properties(MODULE, "src (top level)", "main @ abc", "f00")
    assert props["File"]["title"][0]["text"]["content"] == "src/a.py"
    assert props["Folder"]["select"]["name"] == "src (top level)"
    assert props["Definitions"]["number"] == 3
    assert props["Source Hash"]["rich_text"][0]["text"]["content"] == "f00"
    assert row_properties(MODULE, "x", "m", "")["Source Hash"]["rich_text"] == []


def test_content_hash_changes_with_the_content_but_not_the_commit():
    changed = {**MODULE, "a": "Different."}
    assert content_hash(MODULE, "f") != content_hash(changed, "f")
    assert content_hash(MODULE, "f") == content_hash(dict(MODULE), "f")


def test_plan_creates_replaces_archives_and_skips():
    rows = [
        Row("1", "src/same.py", "h1"),
        Row("2", "src/changed.py", "old"),
        Row("3", "src/gone.py", "h3"),
        Row("4", "src/unfinished.py", ""),
        Row("5", "src/same.py", "h1"),  # a duplicate left by an interrupted sync
    ]
    wanted = {"src/same.py": "h1", "src/changed.py": "new", "src/unfinished.py": "h4",
              "src/new.py": "h5"}  # fmt: skip
    plan = plan_sync(rows, wanted)
    assert plan.create == ["src/new.py"]
    assert [r.page_id for r in plan.replace] == ["2", "4"]
    assert sorted(r.page_id for r in plan.archive) == ["3", "5"]
    assert plan.unchanged == ["src/same.py"]


class FakeNotion:
    """Records calls and serves a fixed set of rows."""

    def __init__(self, rows):
        self.rows, self.calls, self.next_id = rows, [], 100

    def query_database(self, database_id):
        return self.rows

    def create_row(self, database_id, properties, children):
        self.next_id += 1
        self.calls.append(("create", properties["File"]["title"][0]["text"]["content"]))
        return str(self.next_id)

    def append_blocks(self, page_id, children):
        self.calls.append(("append", page_id))

    def update_properties(self, page_id, properties):
        self.calls.append(("hash", page_id))

    def archive_page(self, page_id):
        self.calls.append(("archive", page_id))


def api_row(page_id, path, digest):
    return {"id": page_id, "properties": {
        "File": {"title": [{"plain_text": path}]},
        "Source Hash": {"rich_text": [{"plain_text": digest}] if digest else []},
    }}  # fmt: skip


def test_sync_writes_the_new_row_before_archiving_the_old_one():
    data = {"meta": "main @ abc", "pkgs": [{"name": "src (top level)", "mods": [MODULE]}]}
    fake = FakeNotion([api_row("old", "src/a.py", "stale")])
    plan = sync(fake, "db", data)
    assert [r.page_id for r in plan.replace] == ["old"]
    assert fake.calls == [("create", "src/a.py"), ("hash", "101"), ("archive", "old")]


def test_sync_skips_unchanged_rows_and_a_dry_run_writes_nothing():
    data = {"meta": "main @ abc", "pkgs": [{"name": "src (top level)", "mods": [MODULE]}]}
    digest = content_hash(MODULE, "src (top level)")
    fake = FakeNotion([api_row("p", "src/a.py", digest)])
    assert sync(fake, "db", data).unchanged == ["src/a.py"] and fake.calls == []
    fake = FakeNotion([])
    assert sync(fake, "db", data, dry_run=True).create == ["src/a.py"] and fake.calls == []


def scripted(responses):
    """A send function that replays (status, headers, body) and records the requests."""
    sent = []

    def send(method, url, headers, body):
        sent.append((method, url, json.loads(body) if body else None))
        status, hdrs, payload = responses.pop(0)
        return status, hdrs, json.dumps(payload).encode()

    return send, sent


def test_client_retries_rate_limits_after_the_requested_delay():
    send, sent = scripted([(429, {"Retry-After": "2"}, {}), (200, {}, {"id": "p1"})])
    waits = []
    client = NotionClient("secret", send=send, sleep=waits.append)
    assert client.create_row("db", {}, []) == "p1"
    assert waits == [2.0] and len(sent) == 2


def test_client_follows_pagination():
    send, sent = scripted([
        (200, {}, {"results": [{"id": "a"}], "has_more": True, "next_cursor": "c1"}),
        (200, {}, {"results": [{"id": "b"}], "has_more": False}),
    ])  # fmt: skip
    rows = list(NotionClient("secret", send=send).query_database("db"))
    assert [r["id"] for r in rows] == ["a", "b"]
    assert sent[1][2]["start_cursor"] == "c1"


def test_client_errors_carry_the_api_message_and_never_the_token():
    send, _ = scripted([(404, {}, {"code": "object_not_found", "message": "Not shared."})])
    with pytest.raises(NotionError, match="object_not_found: Not shared.") as err:
        NotionClient("secret-token", send=send).archive_page("p")
    assert "secret-token" not in str(err.value)


def test_client_without_a_token_fails_loudly():
    with pytest.raises(NotionError, match="NOTION_TOKEN"):
        NotionClient("")
