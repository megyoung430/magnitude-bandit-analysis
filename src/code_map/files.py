"""Reading the repo and the hand-written inputs of the code map (the I/O edge)."""

import json
import subprocess
import tomllib
from pathlib import Path

from src.code_map.assemble import Overviews, overviews_from_toml, parse_descriptions
from src.code_map.extract import ModuleRecord, NotebookRecord, module_record, notebook_record

SOURCE_FOLDERS = ("src", "scripts")


def skipped(path: Path, root: Path) -> bool:
    """True for files inside caches and hidden folders (``__pycache__``, ``.ipynb_checkpoints``)."""
    parts = path.relative_to(root).parts[:-1]
    return any(p == "__pycache__" or p.startswith(".") for p in parts)


def source_paths(root: Path) -> tuple[list[Path], list[Path]]:
    """The Python files and notebooks under ``src/`` and ``scripts/``, sorted by path."""
    root = Path(root)
    python, notebooks = [], []
    for folder in SOURCE_FOLDERS:
        python += [p for p in (root / folder).rglob("*.py") if not skipped(p, root)]
        notebooks += [p for p in (root / folder).rglob("*.ipynb") if not skipped(p, root)]
    return sorted(python), sorted(notebooks)


def read_records(root: Path) -> tuple[list[ModuleRecord], list[NotebookRecord]]:
    """Every Python file and notebook of the repo at *root*, read and parsed."""
    root = Path(root)
    python, notebooks = source_paths(root)

    def name(p: Path) -> str:
        return p.relative_to(root).as_posix()

    modules = [module_record(name(p), p.read_text(encoding="utf-8")) for p in python]
    books = [notebook_record(name(p), json.loads(p.read_text(encoding="utf-8"))) for p in notebooks]
    return modules, books


def load_overviews(path: Path) -> Overviews:
    """The hand-written page text from ``overviews.toml``."""
    with open(path, "rb") as f:
        return overviews_from_toml(tomllib.load(f))


def load_descriptions(path: Path) -> dict[str, str]:
    """Written descriptions from ``descriptions.tsv``."""
    return parse_descriptions(Path(path).read_text(encoding="utf-8"))


def git_meta(root: Path) -> str:
    """Branch, short commit and commit date of *root*, as "<branch> @ <commit>, <date>".

    Raises:
        RuntimeError: If *root* is not a git checkout with at least one commit.
    """
    try:
        branch = git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
        commit, date = git_output(root, "log", "-1", "--format=%h %cs").split()
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as err:
        raise RuntimeError(f"cannot read the git commit of {root}: {err}") from err
    return f"{branch} @ {commit}, {date}"


def git_output(root: Path, *args: str) -> str:
    """stdout of one git command run in *root*."""
    done = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def write_page(path: Path, html: str) -> None:
    """Write the page, creating its folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
