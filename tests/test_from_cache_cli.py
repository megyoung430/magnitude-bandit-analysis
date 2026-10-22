"""``--from-cache`` on the two pooled-run scripts."""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "full_pipelines"
HELP = "sessions added since then are missing"


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


def test_run_cohorts_reads_from_cache_with_the_paper_ready_command(cohorts_cli):
    assert cohorts_cli.parse_args([]).from_cache is False
    args = cohorts_cli.parse_args(
        ["--cohorts", "3", "--pooled-only", "--paper-ready", "--from-cache"]
    )
    assert args.from_cache and args.pooled_only and args.paper_ready


def test_run_cohorts_passes_from_cache_on(cohorts_cli, monkeypatch):
    seen = recorder(monkeypatch, cohorts_cli, "run_cohorts")
    cohorts_cli.main(["--cohorts", "3", "--pooled-only", "--paper-ready", "--from-cache"])
    assert seen["from_cache"] is True
    cohorts_cli.main(["--cohorts", "3", "--pooled-only"])
    assert seen["from_cache"] is False


def test_all_drifting_reads_from_cache(all_cli, monkeypatch):
    assert all_cli.parse_args(["--cohorts", "1"]).from_cache is False
    seen = recorder(monkeypatch, all_cli, "run_all_cohorts_drifting")
    all_cli.main(["--cohorts", "1", "3", "4", "--paper-ready", "--from-cache"])
    assert (seen["from_cache"], seen["use_cache"], seen["paper_ready"]) == (True, True, True)


def test_from_cache_and_no_cache_exclude_each_other(all_cli):
    with pytest.raises(SystemExit):
        all_cli.parse_args(["--cohorts", "1", "--from-cache", "--no-cache"])


@pytest.mark.parametrize("script", ["run_cohorts", "run_all_drifting_problems"])
def test_the_help_says_what_is_missing(script, capsys):
    with pytest.raises(SystemExit):
        load_script(script).parse_args(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--from-cache" in help_text and HELP in help_text
