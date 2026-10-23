"""The pooled runs with ``from_cache=True``: same outputs, no share needed.

The cache is filled by a normal run on a raw folder whose import is replaced by the synthetic
cohorts (writing real pyControl files is not needed to exercise the cache), then the raw folder
is renamed away, as when the share is unmounted, and the run is repeated from the cache.
"""

import builtins
import hashlib
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

import src.behavior_import.load_trials as load_trials
import src.settings as settings
from src.behavior_import.import_log import ImportLog
from src.behavior_import.load_trials import ImportedTrials, StaleCacheWarning
from src.pipelines.all_cohorts_drifting import run_all_cohorts_drifting
from src.pipelines.cache_provenance import FROM_CACHE_FILE
from src.pipelines.cohort_analysis import cohort_raw_dir, run_cohorts
from tests.all_cohorts_fixtures import synthetic_cohort
from tests.cohort_fixtures import blocked_then_two_drifting

pytestmark = pytest.mark.filterwarnings("ignore")

COHORTS = {
    1: synthetic_cohort(("a1", "a2", "a3"), seed=1),
    3: blocked_then_two_drifting(),
}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def digests(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of the CSV and PNG files, but not the warnings log."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.suffix in {".csv", ".png"} and p.name != "warnings_log.csv"
    }


@pytest.fixture(scope="module")
def share(tmp_path_factory):
    """``(data root, unmount)``: a share with cohorts 1 and 3, a cache, and a way to lose it."""
    base = tmp_path_factory.mktemp("from-cache")
    data_root = base / "share"
    with pytest.MonkeyPatch.context() as patch:

        def fake_import(root, classify_task_types=False, parse=None, paths=()):
            for path in paths:  # as the import does, so the cache sees every file read
                parse(path)
            cohort_id = int(Path(root).name.removeprefix("cohort-"))
            return ImportedTrials(COHORTS[cohort_id], ImportLog())

        patch.setattr(load_trials, "_import_and_extract", fake_import)
        patch.setenv("BANDIT_DATA_ROOT", str(data_root))
        patch.setenv("BANDIT_CACHE_DIR", str(base / "cache"))
        for cohort_id in COHORTS:
            raw = cohort_raw_dir(data_root, cohort_id) / "sub-01_id-M1" / "ses-01" / "behav"
            raw.mkdir(parents=True)
            (raw / "M1.tsv").write_text("time\ttype\tsubtype\tcontent\n")
        settings.get_settings.cache_clear()
        settings.get_unchecked_settings.cache_clear()
        yield base, data_root
        settings.get_settings.cache_clear()
        settings.get_unchecked_settings.cache_clear()


def run_cohort_3(results_root: Path, **kwargs) -> None:
    run_cohorts(
        [3],
        pooled_only=True,
        exclude_mice=["m2"],
        results_root=results_root,
        **kwargs,
    )


def run_pool(results_root: Path, **kwargs) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_all_cohorts_drifting([1, 3], results_root=results_root, exclude_mice=["m2"], **kwargs)


@contextmanager
def file_system_spy(prefix: str) -> Iterator[list]:
    """Record every os-level access to a path under *prefix* made inside the block."""
    touched = []

    def path_of(p) -> str:
        try:
            s = os.fspath(p)
        except TypeError:
            return ""
        return s.decode() if isinstance(s, bytes) else s

    def spy(name, original):
        def wrapper(p, *args, **kwargs):
            if path_of(p).startswith(prefix):
                touched.append((name, path_of(p)))
            return original(p, *args, **kwargs)

        return wrapper

    with pytest.MonkeyPatch.context() as patch:
        for name in ("stat", "lstat", "listdir", "scandir", "access"):
            patch.setattr(os, name, spy(name, getattr(os, name)))
        patch.setattr(builtins, "open", spy("open", builtins.open))
        yield touched


@pytest.fixture(scope="module")
def runs(share):
    base, data_root = share
    roots = {
        kind: base / kind for kind in ("cohort-share", "cohort-cache", "all-share", "all-cache")
    }
    run_cohort_3(roots["cohort-share"])
    run_pool(roots["all-share"])
    data_root.rename(base / "unmounted")
    with file_system_spy(str(data_root)) as touched:
        run_cohort_3(roots["cohort-cache"], from_cache=True)
        run_cohort_3(roots["cohort-cache"], from_cache=True, dry_run=True)
        run_pool(roots["all-cache"], from_cache=True)
    roots["touched"] = touched
    return roots


def test_the_from_cache_runs_make_no_os_level_access_to_the_share(runs):
    assert runs["touched"] == []


def test_the_cohort_pooled_run_is_the_same_from_the_cache(runs):
    share_run, cache_run = digests(runs["cohort-share"]), digests(runs["cohort-cache"])
    assert any("all-drifting-problems-without-m2" in p for p in share_run)
    assert cache_run == share_run


def test_the_all_cohorts_run_is_the_same_from_the_cache(runs):
    share_run, cache_run = digests(runs["all-share"]), digests(runs["all-cache"])
    assert any("all-cohorts-without-m2" in p for p in share_run)
    assert cache_run == share_run


def test_the_cohort_warnings_log_records_the_stale_cache(runs):
    log = pd.read_csv(next(runs["cohort-cache"].rglob("warnings_log.csv")))
    assert "StaleCacheWarning" in set(log["category"])
    plain = pd.read_csv(next(runs["cohort-share"].rglob("warnings_log.csv")))
    assert "StaleCacheWarning" not in set(plain["category"])


def test_without_from_cache_the_unmounted_share_still_fails_loudly(runs, share, tmp_path):
    _, data_root = share
    with pytest.raises(FileNotFoundError, match="is the share mounted"):
        run_cohort_3(tmp_path, data_root=data_root)
    # the settings were read while the share was mounted; read them again now it is gone
    settings.get_settings.cache_clear()
    settings.get_unchecked_settings.cache_clear()
    with pytest.raises(ValidationError, match="data_root"):
        run_cohort_3(tmp_path)


def test_from_cache_runs_with_import_code_changed_since_the_cache(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(load_trials, "import_code_version", lambda: "edited since")
    with pytest.warns(StaleCacheWarning, match="older import code"):
        run_cohort_3(tmp_path, from_cache=True)
    assert digests(tmp_path) == digests(runs["cohort-cache"])


# --- provenance ---------------------------------------------------------------------------

COHORT_NOTE_FOLDERS = [
    "figures/open-field/cohort-03",
    "tables/open-field/cohort-03",
]
ALL_NOTE_FOLDERS = [
    "figures/open-field/all-cohorts",
    "figures/open-field/all-cohorts-without-m2",
    "tables/open-field/all-cohorts",
    "tables/open-field/all-cohorts-without-m2",
]


@pytest.mark.parametrize(
    ("run", "folders"), [("cohort", COHORT_NOTE_FOLDERS), ("all", ALL_NOTE_FOLDERS)]
)
def test_from_cache_runs_say_so_in_each_output_folder(runs, run, folders):
    for folder in folders:
        note = runs[f"{run}-cache"] / folder / FROM_CACHE_FILE
        assert "cohort-03: cached import of" in note.read_text(), folder
        assert not (runs[f"{run}-share"] / folder / FROM_CACHE_FILE).exists(), folder
    assert len(list(runs[f"{run}-cache"].rglob(FROM_CACHE_FILE))) == len(folders)


def test_the_all_cohorts_note_lists_both_cohorts(runs):
    text = (runs["all-cache"] / ALL_NOTE_FOLDERS[0] / FROM_CACHE_FILE).read_text()
    assert "cohort-01: cached import of" in text and "cohort-03: cached import of" in text
