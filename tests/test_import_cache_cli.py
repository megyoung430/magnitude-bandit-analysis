"""``scripts/full_pipelines/import_cache.py``: check, refresh and prune from the command line."""

import importlib.util
import json
from pathlib import Path

import pytest

import src.settings as settings
from src.behavior_import.cache_store import list_snapshots
from src.pipelines.cohort_analysis import cohort_raw_dir, load_cohort
from tests.test_import_cache_refresh import first_cohort, grow

pytestmark = pytest.mark.filterwarnings("ignore")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "full_pipelines" / "import_cache.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("import_cache_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def share(tmp_path, monkeypatch):
    """``(cohort 3's raw folder, cache dir)`` with the settings pointing at them."""
    data_root, cache = tmp_path / "share", tmp_path / "cache"
    raw = first_cohort(cohort_raw_dir(data_root, 3), unreadable=False)
    monkeypatch.setenv("BANDIT_DATA_ROOT", str(data_root))
    monkeypatch.setenv("BANDIT_CACHE_DIR", str(cache))
    settings.get_settings.cache_clear()
    settings.get_unchecked_settings.cache_clear()
    yield raw, cache
    settings.get_settings.cache_clear()
    settings.get_unchecked_settings.cache_clear()


def test_check_refresh_check_round_trip(cli, share, capsys):
    raw, cache = share
    assert cli.main(["check", "--cohorts", "3"]) == cli.NOT_UP_TO_DATE
    out = capsys.readouterr().out
    assert "cohort-03: no import cache of" in out
    assert "import_cache.py refresh --cohorts 3" in out

    assert cli.main(["refresh", "--cohorts", "3"]) == 0
    out = capsys.readouterr().out
    assert "cohort-03: refreshed; 5 file(s) parsed, 0 reused" in out
    assert len(list_snapshots(cache).snapshots) == 1

    assert cli.main(["check", "--cohorts", "3"]) == cli.UP_TO_DATE
    assert "up to date" in capsys.readouterr().out

    grow(raw)
    assert cli.main(["check", "--cohorts", "3"]) == cli.NOT_UP_TO_DATE
    out = capsys.readouterr().out
    assert "M2 (new subject)" in out and "removed  M1-2026-01-01-110000.tsv" in out

    assert cli.main(["refresh", "--cohorts", "3"]) == 0
    assert "4 file(s) parsed, 3 reused" in capsys.readouterr().out
    assert cli.main(["check", "--cohorts", "3"]) == cli.UP_TO_DATE


def test_check_as_json(cli, share, capsys):
    raw, _ = share
    cli.main(["refresh", "--cohorts", "3"])
    capsys.readouterr()
    grow(raw)
    assert cli.main(["check", "--cohorts", "3", "--json"]) == cli.NOT_UP_TO_DATE
    (report,) = json.loads(capsys.readouterr().out)
    assert report["cohort"] == "cohort-03" and report["up_to_date"] is False
    assert report["cache"]["kind"] == "snapshot" and report["cache"]["files"] == 5
    assert (len(report["new"]), len(report["changed"]), len(report["removed"])) == (3, 1, 1)
    assert report["new_subjects"] == ["M2"]


def test_prune_dry_run_deletes_nothing(cli, share, capsys):
    raw, cache = share
    cli.main(["refresh", "--cohorts", "3", "--keep-old"])
    cli.main(["refresh", "--cohorts", "3", "--keep-old"])  # up to date: no second snapshot
    grow(raw)
    cli.main(["refresh", "--cohorts", "3", "--keep-old"])
    before = sorted(p for p in cache.rglob("*"))
    capsys.readouterr()
    assert cli.main(["prune", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would remove snapshot" in out and "MB" in out
    assert sorted(p for p in cache.rglob("*")) == before
    assert cli.main(["prune", "--cohorts", "3"]) == 0
    assert len(list_snapshots(cache).snapshots) == 1


def test_a_missing_share_is_an_error_line_and_exit_code_2(cli, share, capsys):
    raw, cache = share
    raw.rename(raw.parent / "gone")
    assert cli.main(["check", "--cohorts", "3"]) == cli.SERVER_ERROR
    assert "is the share mounted" in capsys.readouterr().err
    assert not cache.exists()


def test_check_and_refresh_need_cohorts(cli):
    with pytest.raises(SystemExit):
        cli.parse_args(["check"])
    with pytest.raises(SystemExit):
        cli.parse_args(["refresh"])
    assert cli.parse_args(["prune"]).cohorts is None


def test_refresh_cache_then_run_uses_the_new_data_and_later_runs_read_it(share, capsys):
    raw, cache = share
    first = load_cohort(3, None, True, True)  # cache-first with no cache: imports and caches
    assert first.log.cache_source is None
    grow(raw)
    stale = load_cohort(3, None, True, True)
    assert stale.log.cache_source is not None
    assert "M2" not in str(stale.trials_by_problem)
    refreshed = load_cohort(3, None, True, False)  # what --refresh-cache runs
    assert "M2" in str(refreshed.trials_by_problem) and refreshed.log.cache_source is None
    (snapshot,) = list_snapshots(cache).snapshots
    later = load_cohort(3, None, True, True)
    assert later.log.cache_source.time == snapshot.manifest.created
    assert "M2" in str(later.trials_by_problem)


def test_missing_settings_are_one_error_line_and_exit_code_2(cli, monkeypatch, capsys):
    monkeypatch.delenv("BANDIT_DATA_ROOT", raising=False)
    monkeypatch.setattr(settings.UncheckedSettings, "model_config", {"env_prefix": "BANDIT_"})
    settings.get_unchecked_settings.cache_clear()
    try:
        assert cli.main(["check", "--cohorts", "3"]) == cli.SERVER_ERROR
    finally:
        settings.get_unchecked_settings.cache_clear()
    err = capsys.readouterr().err
    assert err.startswith("error: the settings are not usable (BANDIT_DATA_ROOT)")
    assert len(err.strip().splitlines()) == 1
