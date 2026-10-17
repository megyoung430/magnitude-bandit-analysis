"""Tests for pairing .ppd recordings with pyControl files and putting them on pyControl time."""

import dataclasses

import numpy as np
import pytest

from src.photometry_import.recordings import (
    PhotometryRecording,
    load_photometry_recordings,
    match_ppd_to_tsv,
)
from src.photometry_import.sessions import find_photometry_sessions
from src.sync.pycontrol_time import align_to_pycontrol, sync_quality
from tests.photometry_fixtures import (
    SAMPLING_RATE,
    VOLTS_PER_DIVISION,
    pulse_trace,
    tsv_table,
    write_ppd,
    write_simple_ppd,
    write_tsv,
)
from tests.sync_fixtures import sample_clock_pulses

SUBJECT = "RT_64_R"


def _pulses(seed, n_pulse=60, offset_s=12.3):
    """Pulse times (ms) on a recorder, and (s) on pyControl; the recorder started at offset_s."""
    return sample_clock_pulses(
        np.random.default_rng(seed), n_pulse=n_pulse, clock_offset_s=offset_s
    )


def _n_samples(pulses_ms, extra=200):
    return int(pulses_ms[-1] * SAMPLING_RATE / 1000) + extra


def _folder(tmp_path, subject=SUBJECT, ses="ses-44_date-20261014"):
    folder = tmp_path / f"sub-06_id-{subject}" / ses
    (folder / "photometry").mkdir(parents=True)
    (folder / "behav").mkdir()
    return folder


def _add_ppd(folder, name, pulses_ms, subject=SUBJECT):
    path = folder / "photometry" / name
    write_simple_ppd(
        path, subject, np.round(pulses_ms * SAMPLING_RATE / 1000), _n_samples(pulses_ms)
    )
    return path


def _add_tsv(folder, name, pulses_s):
    return write_tsv(folder / "behav" / name, tsv_table(pulses_s))


def _load(tmp_path, **kwargs):
    (session,) = find_photometry_sessions(tmp_path)
    return load_photometry_recordings(session, **kwargs)


# --- match_ppd_to_tsv ---------------------------------------------------------------------


def test_match_picks_the_tsv_whose_pulses_align():
    ppd_ms, tsv_s = _pulses(1)
    _, other_a = _pulses(2)
    _, other_b = _pulses(3)
    assert match_ppd_to_tsv(ppd_ms, [other_a, tsv_s, other_b]) == 1


def test_match_is_none_when_no_tsv_aligns():
    ppd_ms, _ = _pulses(1)
    assert match_ppd_to_tsv(ppd_ms, [_pulses(2)[1], _pulses(3)[1]]) is None


def test_match_is_none_for_a_ppd_with_too_few_pulses():
    ppd_ms, tsv_s = _pulses(1, n_pulse=60)
    assert match_ppd_to_tsv(ppd_ms[:12], [tsv_s]) is None
    assert match_ppd_to_tsv(np.array([]), [tsv_s]) is None


def test_match_min_pulses_is_an_argument():
    ppd_ms, tsv_s = _pulses(1, n_pulse=60)
    assert match_ppd_to_tsv(ppd_ms[:30], [tsv_s], min_pulses=40) is None
    assert match_ppd_to_tsv(ppd_ms[:30], [tsv_s], min_pulses=20) == 0


def test_match_skips_tsvs_too_short_to_align():
    ppd_ms, tsv_s = _pulses(1)
    assert match_ppd_to_tsv(ppd_ms, [np.array([]), tsv_s[:5], tsv_s]) == 2


def test_match_raises_when_two_tsvs_align():
    ppd_ms, tsv_s = _pulses(1)
    with pytest.raises(ValueError, match="more than one"):
        match_ppd_to_tsv(ppd_ms, [tsv_s, tsv_s + 1000.0])


# --- load_photometry_recordings -----------------------------------------------------------


def test_recording_carries_signals_on_pycontrol_time(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, tsv_s = _pulses(1)
    ppd = _add_ppd(folder, "a.ppd", ppd_ms)
    tsv = _add_tsv(folder, "a.tsv", tsv_s)

    recordings, skipped = _load(tmp_path)

    assert skipped == []
    (rec,) = recordings
    assert isinstance(rec, PhotometryRecording)
    assert (rec.subject, rec.session) == (SUBJECT, "ses-44")
    assert (rec.ppd_path, rec.tsv_path) == (ppd, tsv)
    assert rec.fs == SAMPLING_RATE
    n = _n_samples(ppd_ms)
    for array in (rec.time, rec.signal, rec.control, rec.signal_led_on, rec.signal_baseline):
        assert array.shape == (n,)
    # LED-off baseline is subtracted: 5000 - 1000 counts for signal, 2500 - 1000 for control.
    np.testing.assert_allclose(rec.signal, 4000 * VOLTS_PER_DIVISION)
    np.testing.assert_allclose(rec.control, 1500 * VOLTS_PER_DIVISION)
    np.testing.assert_allclose(rec.signal_led_on, 5000 * VOLTS_PER_DIVISION)
    np.testing.assert_allclose(rec.control_baseline, 1000 * VOLTS_PER_DIVISION)
    assert not rec.signal_clipping.any()
    assert not rec.control_clipping.any()
    assert rec.led_current == (90, 100)
    # The recorder's first sample was taken at pyControl time 12.3 s.
    inside = ~np.isnan(rec.time)
    np.testing.assert_allclose(
        rec.time[inside], 12.3 + np.arange(n)[inside] / SAMPLING_RATE, atol=8e-3
    )


def test_recording_reports_samples_outside_the_synced_stretch(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, tsv_s = _pulses(1)
    _add_ppd(folder, "a.ppd", ppd_ms)
    _add_tsv(folder, "a.tsv", tsv_s)
    (rec,), _ = _load(tmp_path)
    assert rec.n_samples_outside_sync == int(np.isnan(rec.time).sum())
    assert 0 < rec.n_samples_outside_sync < len(rec.time) / 2
    assert rec.sync_quality["n_matched"] >= 50
    assert rec.sync_quality["max_abs_interval_discrepancy_ms"] < 15


def test_recording_is_frozen(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, tsv_s = _pulses(1)
    _add_ppd(folder, "a.ppd", ppd_ms)
    _add_tsv(folder, "a.tsv", tsv_s)
    (rec,), _ = _load(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.fs = 1


def test_a_ppd_without_sync_pulses_is_skipped_with_a_reason(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, tsv_s = _pulses(1)
    short = folder / "photometry" / "a_short.ppd"
    write_simple_ppd(short, SUBJECT, [], 780)  # a 6 s false start
    full = _add_ppd(folder, "b_full.ppd", ppd_ms)
    _add_tsv(folder, "a.tsv", tsv_s)

    recordings, skipped = _load(tmp_path)

    assert [r.ppd_path for r in recordings] == [full]
    assert [s.ppd_path for s in skipped] == [short]
    assert "0 sync pulses" in skipped[0].reason


def test_a_ppd_matching_no_tsv_is_skipped_with_a_reason(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, _ = _pulses(1)
    ppd = _add_ppd(folder, "a.ppd", ppd_ms)
    _add_tsv(folder, "a.tsv", _pulses(2)[1])

    recordings, skipped = _load(tmp_path)

    assert recordings == []
    assert [s.ppd_path for s in skipped] == [ppd]
    assert "no .tsv" in skipped[0].reason


def test_a_session_without_a_tsv_skips_every_ppd(tmp_path):
    folder = _folder(tmp_path)
    ppd = _add_ppd(folder, "a.ppd", _pulses(1)[0])
    recordings, skipped = _load(tmp_path)
    assert recordings == []
    assert [s.ppd_path for s in skipped] == [ppd]
    assert "no .tsv" in skipped[0].reason


def test_two_ppds_sharing_one_tsv_are_both_recorded(tmp_path):
    folder = _folder(tmp_path)
    # One continuous pyControl run of 100 pulses; the photometry system was stopped and
    # restarted halfway, so each .ppd saw half the pulses on its own clock.
    first_ms, tsv_s = _pulses(1, n_pulse=100, offset_s=5.0)
    half_1 = first_ms[:50]
    restart_s = tsv_s[50] - 2.0  # second .ppd began 2 s before its first pulse
    half_2 = (tsv_s[50:] - restart_s) * 1000
    _add_ppd(folder, "a.ppd", half_1)
    _add_ppd(folder, "b.ppd", half_2)
    tsv = _add_tsv(folder, "run.tsv", tsv_s)

    recordings, skipped = _load(tmp_path)

    assert skipped == []
    assert [r.ppd_path.name for r in recordings] == ["a.ppd", "b.ppd"]
    assert {r.tsv_path for r in recordings} == {tsv}
    # Both are on the same pyControl clock: the second starts after the first ends.
    assert np.nanmax(recordings[0].time) < np.nanmin(recordings[1].time)


def test_a_restart_pairs_each_ppd_with_its_own_tsv(tmp_path):
    folder = _folder(tmp_path)
    ppd_1, tsv_1 = _pulses(1)
    ppd_2, tsv_2 = _pulses(2)
    _add_ppd(folder, "a.ppd", ppd_1)
    _add_ppd(folder, "b.ppd", ppd_2)
    tsv_a = _add_tsv(folder, "a.tsv", tsv_1)
    tsv_b = _add_tsv(folder, "b.tsv", tsv_2)

    recordings, skipped = _load(tmp_path)

    assert skipped == []
    assert [r.tsv_path for r in recordings] == [tsv_a, tsv_b]


def test_a_ppd_from_another_subject_raises(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, tsv_s = _pulses(1)
    _add_ppd(folder, "a.ppd", ppd_ms, subject="RT_999_X")
    _add_tsv(folder, "a.tsv", tsv_s)
    with pytest.raises(ValueError, match="RT_999_X"):
        _load(tmp_path)


def test_a_tsv_without_rsync_events_cannot_match_but_does_not_stop_the_session(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, tsv_s = _pulses(1)
    _add_ppd(folder, "a.ppd", ppd_ms)
    _add_tsv(folder, "a_aborted.tsv", [])
    tsv = _add_tsv(folder, "b.tsv", tsv_s)
    (rec,), skipped = _load(tmp_path)
    assert skipped == []
    assert rec.tsv_path == tsv


def test_clipping_flags_are_carried_through(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, tsv_s = _pulses(1)
    n = _n_samples(ppd_ms)
    signal_on = np.full(n, 5000)
    signal_on[30] = 32767
    write_ppd(
        folder / "photometry" / "a.ppd",
        subject_id=SUBJECT,
        signal_on=signal_on,
        signal_baseline=np.full(n, 1000),
        control_on=np.full(n, 2500),
        control_baseline=np.full(n, 1000),
        sync_trace=pulse_trace(n, np.round(ppd_ms * SAMPLING_RATE / 1000)),
    )
    _add_tsv(folder, "a.tsv", tsv_s)
    (rec,), _ = _load(tmp_path)
    assert np.flatnonzero(rec.signal_clipping).tolist() == [30]
    assert not rec.control_clipping.any()


def test_tsv_pulses_can_be_supplied_instead_of_read(tmp_path):
    folder = _folder(tmp_path)
    ppd_ms, tsv_s = _pulses(1)
    _add_ppd(folder, "a.ppd", ppd_ms)
    tsv = folder / "behav" / "never_written.tsv"
    (session,) = find_photometry_sessions(tmp_path)
    session = dataclasses.replace(session, tsv_paths=(tsv,))
    (rec,), _ = load_photometry_recordings(session, tsv_pulses_s={tsv: tsv_s})
    assert rec.tsv_path == tsv


# --- chance fits are not matches ----------------------------------------------------------


def test_a_ppd_sharing_only_a_few_pulses_with_a_long_tsv_is_not_a_match():
    # The aligner accepts this pair on the strength of its one genuinely shared chunk, but the
    # other pulses of the .ppd have no partner.
    ppd_full, tsv_s = _pulses(1, n_pulse=232)
    tail = ppd_full[7] + np.cumsum(np.random.default_rng(100).uniform(100, 1900, 17))
    ppd_ms = np.r_[ppd_full[:8], tail]
    aligner = align_to_pycontrol(ppd_ms, tsv_s, units_ms=1)
    assert sync_quality(aligner)["fraction_matched"] < 0.5
    assert match_ppd_to_tsv(ppd_ms, [tsv_s]) is None


def test_a_ppd_whose_intervals_disagree_with_the_tsv_is_not_a_match():
    rng = np.random.default_rng(4)
    tsv_ms = np.cumsum(rng.uniform(1000, 5000, 60))
    interval_errors = rng.normal(0, 120, 60)  # ms of disagreement per interval
    ppd_ms = np.cumsum(np.diff(tsv_ms, prepend=0.0) + interval_errors)
    tsv_s = 10.0 + tsv_ms / 1000
    aligner = align_to_pycontrol(ppd_ms, tsv_s, units_ms=1)  # the aligner alone accepts it
    assert sync_quality(aligner)["median_abs_interval_discrepancy_ms"] > 50
    assert match_ppd_to_tsv(ppd_ms, [tsv_s]) is None
