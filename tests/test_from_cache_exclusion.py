"""With --exclude-mice, a from-cache run loads each cohort once (the check reuses its load)."""

import pytest

import src.pipelines.cohort_analysis as cohort_analysis
from src.behavior_import.import_log import ImportLog
from src.behavior_import.load_trials import ImportedTrials
from tests.cohort_fixtures import blocked_then_two_drifting


@pytest.fixture
def loads(monkeypatch):
    calls = []
    log = ImportLog()
    trials = blocked_then_two_drifting()

    def fake_load(cohort_id, data_root, use_cache, from_cache=False):
        calls.append((cohort_id, from_cache))
        return ImportedTrials(trials, log)

    monkeypatch.setattr(cohort_analysis, "load_cohort", fake_load)
    return calls, log, trials


@pytest.fixture
def runner_calls(monkeypatch):
    calls = []
    for name in ("run_cohort", "run_cohort_incremental", "plan_cohort_pooled_only"):
        monkeypatch.setattr(
            cohort_analysis,
            name,
            lambda cohort_id, _name=name, **kw: calls.append((_name, cohort_id, kw)) or "plan",
        )
    return calls


def test_a_from_cache_pooled_run_with_excluded_mice_loads_each_cohort_once(loads, runner_calls):
    calls, log, trials = loads
    cohort_analysis.run_cohorts([3], pooled_only=True, exclude_mice=["m2"], from_cache=True)
    assert calls == [(3, True)]
    ((name, cohort_id, kwargs),) = runner_calls
    assert (name, cohort_id) == ("run_cohort", 3)
    # the trials the check loaded, not a second load
    assert kwargs["trials_by_problem"] is trials
    assert kwargs["import_log"] is log


def test_a_from_cache_dry_run_with_excluded_mice_loads_each_cohort_once(loads, runner_calls):
    calls, _, _ = loads
    cohort_analysis.run_cohorts(
        [3], pooled_only=True, dry_run=True, exclude_mice=["m2"], from_cache=True
    )
    assert calls == [(3, True)]
    ((name, _, kwargs),) = runner_calls
    assert name == "plan_cohort_pooled_only" and kwargs["trials_by_problem"]


def test_a_run_from_the_server_still_reloads_after_the_check(loads, runner_calls):
    calls, _, _ = loads
    cohort_analysis.run_cohorts([3], pooled_only=True, exclude_mice=["m2"])
    assert calls == [(3, False)]  # the check's load; run_cohort (replaced here) loads its own
    ((_, _, kwargs),) = runner_calls
    assert "trials_by_problem" not in kwargs
