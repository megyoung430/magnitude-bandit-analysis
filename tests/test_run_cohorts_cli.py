"""The options of ``scripts/full_pipelines/run_cohorts.py`` that select a partial run."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "full_pipelines" / "run_cohorts.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("run_cohorts_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_without_options_the_run_is_a_full_one(cli):
    args = cli.parse_args([])
    assert (args.last_sessions, args.since, args.dry_run) == (None, None, False)


def test_last_sessions_and_since_are_read(cli):
    assert cli.parse_args(["--last-sessions", "4"]).last_sessions == 4
    assert cli.parse_args(["--cohorts", "3", "4", "--since", "2026-10-16"]).since == "2026-10-16"


def test_last_sessions_and_since_exclude_each_other(cli):
    with pytest.raises(SystemExit):
        cli.parse_args(["--last-sessions", "2", "--since", "2026-10-16"])


def test_dry_run_needs_a_selection(cli):
    with pytest.raises(SystemExit):
        cli.parse_args(["--dry-run"])
    assert cli.parse_args(["--dry-run", "--last-sessions", "1"]).dry_run


def test_main_passes_the_selection_to_run_cohorts(cli, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_cohorts", lambda ids, **kw: seen.update(ids=ids, **kw))
    cli.main(["--cohorts", "4", "--since", "2026-10-16", "--dry-run"])
    assert seen["ids"] == [4]
    assert (seen["since"], seen["last_sessions"], seen["dry_run"]) == ("2026-10-16", None, True)
