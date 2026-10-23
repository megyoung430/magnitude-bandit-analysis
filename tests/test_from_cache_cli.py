"""The cache options of the two pooled-run scripts: cache-first by default, --refresh-cache."""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "full_pipelines"
HELP = "sessions added since then are missing until --refresh-cache"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cohorts_cli():
    return load_script("run_cohorts")


@pytest.fixture(scope="module")
def all_cli():
    return load_script("run_all_drifting_problems")


def recorder(monkeypatch, module, name):
    seen = {}
    monkeypatch.setattr(module, name, lambda ids, **kw: seen.update(ids=ids, **kw))
    return seen


def test_run_cohorts_reads_the_cache_by_default(cohorts_cli, monkeypatch):
    seen = recorder(monkeypatch, cohorts_cli, "run_cohorts")
    cohorts_cli.main(["--cohorts", "3", "--pooled-only"])
    assert seen["from_cache"] is True


def test_run_cohorts_reads_the_server_with_refresh_cache(cohorts_cli, monkeypatch):
    seen = recorder(monkeypatch, cohorts_cli, "run_cohorts")
    cohorts_cli.main(["--cohorts", "3", "--pooled-only", "--refresh-cache"])
    assert seen["from_cache"] is False


def test_from_cache_is_still_accepted_and_changes_nothing(cohorts_cli, monkeypatch, capsys):
    seen = recorder(monkeypatch, cohorts_cli, "run_cohorts")
    cohorts_cli.main(["--cohorts", "3", "--pooled-only", "--from-cache"])
    assert seen["from_cache"] is True
    assert "--from-cache is no longer needed" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], (True, True)),
        (["--refresh-cache"], (False, True)),
        (["--no-cache"], (False, False)),
        (["--from-cache"], (True, True)),
    ],
)
def test_all_drifting_cache_options(all_cli, monkeypatch, flags, expected):
    seen = recorder(monkeypatch, all_cli, "run_all_cohorts_drifting")
    all_cli.main(["--cohorts", "1", "3", "4", *flags])
    assert (seen["from_cache"], seen["use_cache"]) == expected
    assert "paper_ready" not in seen


@pytest.mark.parametrize(
    "flags",
    [
        ["--from-cache", "--no-cache"],
        ["--refresh-cache", "--no-cache"],
        ["--refresh-cache", "--from-cache"],
    ],
)
def test_the_cache_options_exclude_each_other(all_cli, flags):
    with pytest.raises(SystemExit):
        all_cli.parse_args(["--cohorts", "1", *flags])


def test_run_cohorts_refresh_and_from_cache_exclude_each_other(cohorts_cli):
    with pytest.raises(SystemExit):
        cohorts_cli.parse_args(["--refresh-cache", "--from-cache"])


@pytest.mark.parametrize("script", ["run_cohorts", "run_all_drifting_problems"])
def test_the_help_says_runs_read_the_cache_and_how_to_refresh(script, capsys):
    with pytest.raises(SystemExit):
        load_script(script).parse_args(["--help"])
    # compared without whitespace: argparse wraps lines at hyphens too
    help_text = "".join(capsys.readouterr().out.split())
    for expected in ("--refresh-cache", HELP, "--from-cache", "deprecated"):
        assert "".join(expected.split()) in help_text


@pytest.mark.parametrize(
    "script, argv",
    [("cohorts_cli", ["--cohorts", "3", "--pooled-only"]), ("all_cli", ["--cohorts", "1", "3"])],
)
def test_neither_script_accepts_paper_ready(script, argv, request):
    cli = request.getfixturevalue(script)
    with pytest.raises(SystemExit):
        cli.parse_args([*argv, "--paper-ready"])
