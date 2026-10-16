"""Synthetic pyControl recordings for the tests of changes logged between recordings.

A recording is a frame of ``time``/``type``/``subtype``/``content`` rows: a ``run_start``
snapshot, one ``print`` row per trial and a ``run_end`` snapshot, in the layout of the real
files (counter names ``num_blocks``, ``tot_num_blocks``, ``num_good_rev``, ...).
"""

import json

import pandas as pd

DRIFTING_TASK = "Meg\\3x3_field_drifting_value_bandit"
BLOCKED_TASK = "Meg\\3x3_field_blocked_reward_bandit"
TOWERS = ("B1", "B3", "C2")

# Drifting blocks: one arm is at 0, the others drift by a step per trial.
ZERO_B3 = {"B1": 4, "B3": 0, "C2": 2}
ZERO_B3_DRIFTED = {"B1": 3, "B3": 0, "C2": 2}
ZERO_B1 = {"B1": 0, "B3": 4, "C2": 3}
ZERO_C2 = {"B1": 2, "B3": 3, "C2": 0}

# Blocked magnitudes: a permutation of (4, 1, 0).
B1_BEST = {"B1": 4, "B3": 1, "C2": 0}
B3_BEST = {"B1": 0, "B3": 4, "C2": 1}  # a good reversal from B1_BEST: the best tower changes
SECOND_SWAPPED = {"B1": 4, "B3": 0, "C2": 1}  # best unchanged: only the two lower arms swap

# What a recording logs in ``run_start`` while it is still finding the ports.
PORT_FINDING_MAGS = {"B1": 2, "B3": 2, "C2": 2}

COUNTERS = ("blocks", "good", "bad")
_NAMES = {
    "blocks": ("num_blocks", "tot_num_blocks"),
    "good": ("num_good_rev", "tot_num_good_rev"),
    "bad": ("num_bad_rev", "tot_num_bad_rev"),
}


def snapshot(mags, *, problem=5, found_ports=None, **counters):
    """A ``run_start``/``run_end`` payload.

    Args:
        mags (dict | None): ``curr_rew_mag``; ``None`` leaves it out.
        problem (int): ``num_problem``.
        found_ports (bool | None): ``found_ports``; ``None`` leaves it out.
        **counters: Per-recording and session-total counters, e.g. ``blocks=1, tot_blocks=2``
            or ``good=0, tot_good=3``; a counter that is not given is not logged.
    """
    content = {"num_problem": problem}
    if found_ports is not None:
        content["found_ports"] = found_ports
    if mags is not None:
        content["curr_rew_mag"] = mags
    for kind, (per_run, total) in _NAMES.items():
        if kind in counters:
            content[per_run] = counters[kind]
        if "tot_" + kind in counters:
            content[total] = counters["tot_" + kind]
    return content


def trial_print(n, mags, *, choice="B1", **counters):
    """One trial's ``print`` payload: per-recording counters only, as the task prints them."""
    content = {
        "num_t": n,
        "curr_rew_mag": mags,
        "chosen_tow": choice,
        "c_tows": list(TOWERS),
        "chosen_rank": "best",
        "rank_counts": {"best": n, "second": 0, "third": 0},
    }
    for kind, (per_run, _) in _NAMES.items():
        if kind in counters:
            content[per_run] = counters[kind]
    return content


def port_finding_start(**counters):
    """The ``run_start`` of a recording that begins by finding the ports."""
    return snapshot(PORT_FINDING_MAGS, found_ports=False, **counters)


def port_finding_prints(n):
    """*n* port-finding prints: ``num_t_found`` and no ``num_t``."""
    return [{"num_t_found": i + 1, "found_ports": False} for i in range(n)]


def recording(start, trials, end=None, *, start_time=0.0):
    """A recording frame; *end* ``None`` leaves out ``run_end`` (a crashed recording)."""
    rows = [(start_time, "variable", "run_start", json.dumps(start))]
    rows += [
        (start_time + 1 + i, "variable", "print", json.dumps(trial))
        for i, trial in enumerate(trials)
    ]
    if end is not None:
        rows.append((start_time + 1 + len(trials), "variable", "run_end", json.dumps(end)))
    return pd.DataFrame(rows, columns=["time", "type", "subtype", "content"])


def prints(mags, n, first=1, **counters):
    """*n* identical trial prints numbered from *first*."""
    return [trial_print(first + i, mags, **counters) for i in range(n)]


def session(frames, task=DRIFTING_TASK):
    """A session dict as the importer leaves it, ready for ``extract_trials``."""
    return {"data": list(frames), "task_name": task}
