"""The pooled paper-ready runs with ``from_cache=True``: same outputs, no share needed.

The cache is filled by a normal run on a raw folder whose import is replaced by the synthetic
cohorts (writing real pyControl files is not needed to exercise the cache), then the raw folder
is renamed away, as when the share is unmounted, and the run is repeated from the cache.
"""

import hashlib
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import src.behavior_import.load_trials as load_trials
import src.settings as settings
from src.behavior_import.import_log import ImportLog
from src.behavior_import.load_trials import ImportedTrials, StaleCacheWarning
from src.behavior_visualization import plot_helpers
from src.pipelines.all_cohorts_drifting import run_all_cohorts_drifting
from src.pipelines.cohort_analysis import cohort_raw_dir, run_cohorts
from tests.all_cohorts_fixtures import synthetic_cohort
from tests.cohort_fixtures import blocked_then_two_drifting

pytestmark = pytest.mark.filterwarnings("ignore")

COHORTS = {
    1: synthetic_cohort(("a1", "a2", "a3"), seed=1),
    3: blocked_then_two_drifting(),
}


@pytest.fixture(scope="module", autouse=True)
def low_resolution():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plot_helpers, "PAPER_DPI", 50)
        yield


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

        def fake_import(root, classify_task_types=False):
            cohort_id = int(Path(root).name.removeprefix("cohort-"))
            return ImportedTrials(COHORTS[cohort_id], ImportLog())

        patch.setattr(load_trials, "_import_and_extract", fake_import)
        patch.setenv("BANDIT_DATA_ROOT", str(data_root))
        patch.setenv("BANDIT_CACHE_DIR", str(base / "cache"))
        for cohort_id in COHORTS:
            raw = cohort_raw_dir(data_root, cohort_id) / "sub-01_id-M1" / "ses-01" / "behav"
            raw.mkdir(parents=True)
            (raw / "M1.tsv").write_text("type\tsubtype\tcontent\n")
        settings.get_settings.cache_clear()
        settings.get_unchecked_settings.cache_clear()
        yield base, data_root
        settings.get_settings.cache_clear()
        settings.get_unchecked_settings.cache_clear()


def run_cohort_3(results_root: Path, **kwargs) -> None:
    run_cohorts(
        [3],
        pooled_only=True,
        paper_ready=True,
        exclude_mice=["m2"],
        results_root=results_root,
        **kwargs,
    )


def run_pool(results_root: Path, **kwargs) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_all_cohorts_drifting([1, 3], results_root=results_root, paper_ready=True, **kwargs)


@pytest.fixture(scope="module")
def runs(share):
    base, data_root = share
    roots = {
        kind: base / kind for kind in ("cohort-share", "cohort-cache", "all-share", "all-cache")
    }
    run_cohort_3(roots["cohort-share"])
    run_pool(roots["all-share"])
    data_root.rename(base / "unmounted")
    run_cohort_3(roots["cohort-cache"], from_cache=True)
    run_pool(roots["all-cache"], from_cache=True)
    return roots


def test_the_cohort_pooled_paper_ready_run_is_the_same_from_the_cache(runs):
    share_run, cache_run = digests(runs["cohort-share"]), digests(runs["cohort-cache"])
    assert any("paper-ready" in p for p in share_run)
    assert any("all-drifting-problems-without-m2" in p for p in share_run)
    assert cache_run == share_run


def test_the_all_cohorts_paper_ready_run_is_the_same_from_the_cache(runs):
    share_run, cache_run = digests(runs["all-share"]), digests(runs["all-cache"])
    assert any("paper-ready" in p for p in share_run)
    assert cache_run == share_run


def test_the_cohort_warnings_log_records_the_stale_cache(runs):
    log = pd.read_csv(next(runs["cohort-cache"].rglob("warnings_log.csv")))
    assert "StaleCacheWarning" in set(log["category"])
    plain = pd.read_csv(next(runs["cohort-share"].rglob("warnings_log.csv")))
    assert "StaleCacheWarning" not in set(plain["category"])


def test_without_from_cache_the_unmounted_share_still_fails_loudly(runs, tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        run_cohort_3(tmp_path)


def test_from_cache_runs_with_import_code_changed_since_the_cache(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(load_trials, "import_code_version", lambda: "edited since")
    with pytest.warns(StaleCacheWarning, match="older import code"):
        run_cohort_3(tmp_path, from_cache=True)
    assert digests(tmp_path) == digests(runs["cohort-cache"])
