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


# --- pooled-only runs and excluded mice ------------------------------------------------------


def test_the_new_options_default_to_off(cli):
    args = cli.parse_args([])
    assert (args.pooled_only, args.exclude_mice) == (False, None)


def test_the_command_for_the_cohort_3_set_without_two_mice_is_read(cli):
    args = cli.parse_args(
        ["--cohorts", "3", "--pooled-only", "--exclude-mice", "MY_175_N", "MY_66_LL"]
    )
    assert args.cohorts == [3] and args.pooled_only
    assert args.exclude_mice == ["MY_175_N", "MY_66_LL"]


@pytest.mark.parametrize("bad", [[], ["../x"], ["MY 1"]])
def test_exclude_mice_needs_plain_mouse_ids(cli, bad):
    with pytest.raises(SystemExit):
        cli.parse_args(["--exclude-mice", *bad])


def test_exclude_mice_also_goes_with_a_full_or_partial_run(cli):
    assert cli.parse_args(["--exclude-mice", "MY_1_A"]).exclude_mice == ["MY_1_A"]
    args = cli.parse_args(["--last-sessions", "3", "--exclude-mice", "MY_1_A", "MY_2_B"])
    assert (args.last_sessions, args.exclude_mice) == (3, ["MY_1_A", "MY_2_B"])


def test_pooled_only_excludes_last_sessions_and_since(cli):
    with pytest.raises(SystemExit):
        cli.parse_args(["--pooled-only", "--last-sessions", "2"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--pooled-only", "--since", "2026-10-16"])


def test_a_pooled_only_dry_run_needs_no_selection(cli):
    assert cli.parse_args(["--pooled-only", "--dry-run"]).dry_run


def test_main_passes_pooled_only_and_the_mice_to_run_cohorts(cli, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_cohorts", lambda ids, **kw: seen.update(ids=ids, **kw))
    cli.main(["--cohorts", "3", "--pooled-only", "--exclude-mice", "MY_175_N", "MY_66_LL"])
    assert seen["ids"] == [3]
    assert seen["pooled_only"] is True
    assert seen["exclude_mice"] == ["MY_175_N", "MY_66_LL"]
    assert seen["last_sessions"] is None and seen["dry_run"] is False


def test_a_bad_mouse_name_is_one_line_not_a_traceback(cli, monkeypatch, capsys):
    from src.behavior_analysis.exclude_mice import ExcludedMiceError

    def fail(ids, **kw):
        raise ExcludedMiceError("cannot exclude MY_9_Z: not among the mice (MY_175_L)")

    monkeypatch.setattr(cli, "run_cohorts", fail)
    with pytest.raises(SystemExit) as stop:
        cli.main(["--cohorts", "3", "--pooled-only", "--exclude-mice", "MY_9_Z"])
    assert stop.value.code != 0
    assert "MY_9_Z" in str(stop.value.code)


# --- paper-ready figures ---------------------------------------------------------------------


def test_paper_ready_is_off_by_default_and_read_with_the_pooled_command(cli):
    assert cli.parse_args([]).paper_ready is False
    args = cli.parse_args(["--cohorts", "3", "--pooled-only", "--paper-ready"])
    assert args.paper_ready and args.pooled_only


def test_main_passes_paper_ready_to_run_cohorts(cli, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_cohorts", lambda ids, **kw: seen.update(ids=ids, **kw))
    cli.main(["--cohorts", "3", "--pooled-only", "--paper-ready", "--exclude-mice", "MY_1_A"])
    assert seen["paper_ready"] is True
    cli.main(["--cohorts", "3", "--pooled-only"])
    assert seen["paper_ready"] is False
