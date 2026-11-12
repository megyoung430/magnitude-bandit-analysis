"""Bring the Notion code-map database in line with the code map (decides what to write).

Each row stores a fingerprint of its content ("Source Hash"). A sync creates rows for new
files, replaces the rows of files whose fingerprint changed, moves rows of deleted files to the
trash, and leaves the rest alone, so a sync after a small change makes few requests. A changed
file's new row is written in full, and only then gets its fingerprint and replaces the old
row, so an interrupted sync never leaves a half-written page that looks current.
"""

from collections.abc import Callable, Iterable
from typing import NamedTuple, Protocol

from src.code_map.notion_blocks import content_hash, page_blocks, request_batches, row_properties


class Client(Protocol):
    """The Notion calls the sync uses (see :class:`src.code_map.notion_api.NotionClient`)."""

    def query_database(self, database_id: str) -> Iterable[dict]: ...
    def create_row(self, database_id: str, properties: dict, children: list[dict]) -> str: ...
    def append_blocks(self, page_id: str, children: list[dict]) -> None: ...
    def update_properties(self, page_id: str, properties: dict) -> None: ...
    def archive_page(self, page_id: str) -> None: ...


class Row(NamedTuple):
    """An existing database row."""

    page_id: str
    path: str
    digest: str


class SyncPlan(NamedTuple):
    """What a sync will do, by file path (``replace`` and ``archive`` hold the old rows)."""

    create: list[str]
    replace: list[Row]
    archive: list[Row]
    unchanged: list[str]


def text_of(prop: dict) -> str:
    """The plain text of a title or rich-text property."""
    items = prop.get("title", prop.get("rich_text", [])) if prop else []
    return "".join(
        item.get("plain_text", item.get("text", {}).get("content", "")) for item in items
    )


def existing_rows(pages: Iterable[dict]) -> list[Row]:
    """Rows of the database from the API's page objects."""
    live = [p for p in pages if not p.get("archived") and not p.get("in_trash")]
    props = [p["properties"] for p in live]
    return [
        Row(p["id"], text_of(q.get("File")), text_of(q.get("Source Hash")))
        for p, q in zip(live, props, strict=True)
    ]


def plan_sync(rows: Iterable[Row], wanted: dict[str, str]) -> SyncPlan:
    """Compare the rows with the wanted ``path → fingerprint``.

    A path with several rows (left by an interrupted sync) keeps one current row at most; the
    others are archived. A row without a fingerprint was never finished and is replaced.
    """
    replace, archive, unchanged = [], [], []
    by_path: dict[str, list[Row]] = {}
    for row in rows:
        by_path.setdefault(row.path, []).append(row)
    for path, found in by_path.items():
        if path not in wanted:
            archive += found
            continue
        current = [r for r in found if r.digest and r.digest == wanted[path]]
        if current:
            unchanged.append(path)
            archive += [r for r in found if r != current[0]]
        else:
            replace.append(found[0])
            archive += found[1:]
    create = sorted(path for path in wanted if path not in by_path)
    return SyncPlan(create, replace, archive, sorted(unchanged))


def files_of(data: dict) -> dict[str, tuple[dict, str]]:
    """``path → (entry, folder)`` for every file and notebook in the code-map data."""
    return {mod["p"]: (mod, pkg["name"]) for pkg in data["pkgs"] for mod in pkg["mods"]}


def write_row(client: Client, database_id: str, mod: dict, folder: str, meta: str) -> str:
    """Create one file's row with all its page content, then set its fingerprint."""
    batches = request_batches(page_blocks(mod, meta))
    page_id = client.create_row(database_id, row_properties(mod, folder, meta, ""), batches[0])
    for batch in batches[1:]:
        client.append_blocks(page_id, batch)
    digest = content_hash(mod, folder)
    done = row_properties(mod, folder, meta, digest)["Source Hash"]
    client.update_properties(page_id, {"Source Hash": done})
    return page_id


def sync(
    client: Client,
    database_id: str,
    data: dict,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = lambda line: None,
) -> SyncPlan:
    """Make the database match *data*.

    Args:
        client: The Notion client.
        database_id: The code-map database.
        data: The code-map data from :func:`src.code_map.assemble.build_map`.
        dry_run: Only plan; write nothing.
        progress: Called with one line per file written.

    Returns:
        The plan that was carried out (or would be, for a dry run).
    """
    files = files_of(data)
    wanted = {path: content_hash(mod, folder) for path, (mod, folder) in files.items()}
    plan = plan_sync(existing_rows(client.query_database(database_id)), wanted)
    if dry_run:
        return plan
    for path in plan.create:
        write_row(client, database_id, *files[path], data["meta"])
        progress(f"created {path}")
    for old in plan.replace:
        write_row(client, database_id, *files[old.path], data["meta"])
        client.archive_page(old.page_id)
        progress(f"replaced {old.path}")
    for old in plan.archive:
        client.archive_page(old.page_id)
        progress(f"archived {old.path or '(untitled row)'}")
    return plan
