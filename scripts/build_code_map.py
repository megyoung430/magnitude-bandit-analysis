"""Build the code map: one searchable HTML page describing every file and function of the repo.

    uv run python scripts/build_code_map.py            # writes results/code_map/
    uv run python scripts/build_code_map.py --strict   # also fails if a definition is undescribed
    uv run python scripts/build_code_map.py --notion   # also syncs the Notion database
    uv run python scripts/build_code_map.py --notion --dry-run   # shows what a sync would change

Descriptions come from docstrings; definitions without one take theirs from
docs/code_map/descriptions.tsv, and the folder, file and notebook summaries come from
docs/code_map/overviews.toml. The script lists definitions that have neither, so they can be
given a docstring (preferred) or a line in the TSV.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.code_map.assemble import build_map, coverage, walk  # noqa: E402
from src.code_map.files import (  # noqa: E402
    git_meta,
    load_descriptions,
    load_notion_database,
    load_overviews,
    notion_token,
    read_records,
    write_page,
)
from src.code_map.notion_api import NotionClient, NotionError  # noqa: E402
from src.code_map.notion_sync import SyncPlan, sync  # noqa: E402
from src.code_map.render import render_html, template_text  # noqa: E402

NOTES = REPO_ROOT / "docs" / "code_map"
DEFAULT_OUT = REPO_ROOT / "results" / "code_map" / "magnitude-bandit-code-map.html"
LISTED = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the page")
    parser.add_argument(
        "--strict", action="store_true", help="exit with an error if a definition is undescribed"
    )
    parser.add_argument(
        "--notion",
        action="store_true",
        help="sync the Notion database of docs/code_map/notion.toml (needs NOTION_TOKEN)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="with --notion: list the changes, write nothing"
    )
    args = parser.parse_args(argv)
    if args.dry_run and not args.notion:
        parser.error("--dry-run only applies with --notion")
    return args


def report(missing: list[str], stale: list[str]) -> None:
    """Print the definitions without a description and the descriptions without a definition."""
    if missing:
        print(f"{len(missing)} definitions have no docstring and no line in descriptions.tsv:")
        print("\n".join(f"  {key}" for key in missing[:LISTED]))
        if len(missing) > LISTED:
            print(f"  ... and {len(missing) - LISTED} more")
    if stale:
        print(
            f"{len(stale)} lines of descriptions.tsv match no definition in this checkout "
            "(renamed, removed, or on another branch):"
        )
        print("\n".join(f"  {key}" for key in stale[:LISTED]))


def report_sync(plan: SyncPlan, dry_run: bool) -> None:
    """Print what the Notion sync did (or would do)."""
    verb = "would be" if dry_run else "were"
    print(
        f"Notion: {len(plan.create)} rows {verb} created, {len(plan.replace)} replaced, "
        f"{len(plan.archive)} moved to the trash; {len(plan.unchanged)} unchanged."
    )


def main(argv: list[str] | None = None) -> int:
    """Build the page; return 1 under --strict when a definition is undescribed."""
    args = parse_args(argv)
    modules, notebooks = read_records(REPO_ROOT)
    descriptions = load_descriptions(NOTES / "descriptions.tsv")
    data = build_map(
        modules, notebooks, load_overviews(NOTES / "overviews.toml"), descriptions,
        git_meta(REPO_ROOT),
    )  # fmt: skip
    write_page(args.out, render_html(data, template_text()))
    n_defs = sum(1 for m in modules for _ in walk(m.defs))
    print(
        f"Wrote {args.out}: {len(modules)} files, {len(notebooks)} notebooks, {n_defs} definitions."
    )
    missing, stale = coverage(modules, descriptions)
    report(missing, stale)
    if args.notion:
        try:
            client = NotionClient(notion_token())
            database = load_notion_database(NOTES / "notion.toml")
            plan = sync(client, database, data, dry_run=args.dry_run, progress=print)
        except NotionError as err:
            # rows written before the error are complete; rerunning finishes the rest
            print(f"Notion sync stopped: {err}", file=sys.stderr)
            return 2
        report_sync(plan, args.dry_run)
    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    sys.exit(main())
