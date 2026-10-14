"""get_bad_reversal_info on synthetic sessions: windows, filters, skips and warnings."""

import copy

from src.behavior_analysis.get_bad_reversal_info import get_bad_reversal_info

TOWERS = ("A", "B", "C")


def make_session(bad, rewards=None, blocks=None, choices=None, ranks=None):
    """One session; *rewards* maps tower -> per-trial magnitudes (default 4/1/0 throughout)."""
    n = len(bad)
    rewards = rewards or {"A": [4] * n, "B": [1] * n, "C": [0] * n}
    choices = choices or {t: [int(t == "A")] * n for t in rewards}
    ranks = ranks or {"best": [1] * n, "second": [0] * n}
    return {
        "trial": list(range(1, n + 1)),
        "blocks": blocks if blocks is not None else [1 + i // 4 for i in range(n)],
        "bad_reversals": list(bad),
        "reward_magnitudes_by_tower": rewards,
        "choices_by_tower": choices,
        "choices_by_rank": ranks,
    }


def one_subject(session):
    return {"mouse": {"ses-1": session}}


# Reversals at trials 3 and 7 of 10.
BAD = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]


def test_reversal_record_holds_every_field():
    rewards = {"A": [4] * 10, "B": [1] * 10, "C": [0] * 10}
    rewards["B"][2], rewards["C"][2] = 0, 1  # trial before reversal@3: still a permutation
    rewards["A"][3] = 1  # reversal trial itself looks different
    choices = {t: list(range(i, i + 10)) for i, t in enumerate(TOWERS)}
    ranks = {"best": list(range(20, 30))}
    session = make_session(BAD, rewards=rewards, choices=choices, ranks=ranks)

    first = get_bad_reversal_info(one_subject(session), pre=2, post=2)["mouse"][0]

    assert first == {
        "reversal_idx": 3,
        "bad_reversal_number": 1,
        "block_id": 1,
        "towers": ["A", "B", "C"],
        "trial_window_idx": {"pre": [1, 2], "post": [3, 4]},
        "reward_magnitudes_by_tower_before": {"A": 4, "B": 0, "C": 1},
        "reward_magnitudes_by_tower_after": {"A": 1, "B": 1, "C": 0},
        "choices_by_tower": {
            "A": {"pre": [1, 2], "post": [3, 4]},
            "B": {"pre": [2, 3], "post": [4, 5]},
            "C": {"pre": [3, 4], "post": [5, 6]},
        },
        "choices_by_rank": {"best": {"pre": [21, 22], "post": [23, 24]}},
    }


def test_default_post_window_ends_at_next_reversal_or_last_trial():
    reversals = get_bad_reversal_info(one_subject(make_session(BAD)))["mouse"]
    assert [r["reversal_idx"] for r in reversals] == [3, 7]
    assert [r["bad_reversal_number"] for r in reversals] == [1, 2]
    assert [r["trial_window_idx"]["post"] for r in reversals] == [[3, 4, 5, 6], [7, 8, 9]]


def test_pre_window_is_clipped_at_session_start():
    first = get_bad_reversal_info(one_subject(make_session(BAD)), pre=50)["mouse"][0]
    assert first["trial_window_idx"]["pre"] == [0, 1, 2]


def test_pre_zero_gives_empty_pre_window():
    first = get_bad_reversal_info(one_subject(make_session(BAD)), pre=0)["mouse"][0]
    assert first["trial_window_idx"]["pre"] == []
    assert first["choices_by_tower"]["A"]["pre"] == []


def test_explicit_post_is_clipped_at_last_trial_but_not_at_next_reversal():
    reversals = get_bad_reversal_info(one_subject(make_session(BAD)), post=6)["mouse"]
    assert reversals[0]["trial_window_idx"]["post"] == [3, 4, 5, 6, 7, 8]  # crosses reversal@7
    assert reversals[1]["trial_window_idx"]["post"] == [7, 8, 9]  # clipped at n_trials


def test_reversal_on_the_last_trial_has_a_one_trial_post_window():
    bad = [0, 0, 0, 1]
    only = get_bad_reversal_info(one_subject(make_session(bad)))["mouse"][0]
    assert only["trial_window_idx"] == {"pre": [0, 1, 2], "post": [3]}


def test_include_first_block_prepends_trial_zero_with_matching_before_and_after():
    reversals = get_bad_reversal_info(one_subject(make_session(BAD)), include_first_block=True)[
        "mouse"
    ]
    first = reversals[0]
    assert [r["reversal_idx"] for r in reversals] == [0, 3, 7]
    assert first["trial_window_idx"] == {"pre": [], "post": [0, 1, 2]}
    assert first["reward_magnitudes_by_tower_before"] == first["reward_magnitudes_by_tower_after"]


def test_include_first_block_ignored_when_counter_does_not_start_at_zero():
    bad = [1, 1, 1, 2, 2]
    reversals = get_bad_reversal_info(one_subject(make_session(bad)), include_first_block=True)[
        "mouse"
    ]
    assert [r["reversal_idx"] for r in reversals] == [3]


def test_reward_pattern_is_a_multiset_match_in_any_tower_order(capsys):
    rewards = {"A": [0] * 10, "B": [4] * 10, "C": [1] * 10}
    reversals = get_bad_reversal_info(one_subject(make_session(BAD, rewards=rewards)))["mouse"]
    assert [r["reversal_idx"] for r in reversals] == [3, 7]
    assert capsys.readouterr().out == ""


def test_reversal_with_wrong_reward_pattern_is_skipped_and_reported(capsys):
    rewards = {"A": [4] * 10, "B": [1] * 10, "C": [0] * 10}
    rewards["C"][2] = 1  # before reversal@3 the rewards are (4, 1, 1)
    reversals = get_bad_reversal_info(one_subject(make_session(BAD, rewards=rewards)))["mouse"]
    assert [r["reversal_idx"] for r in reversals] == [7]
    assert capsys.readouterr().out == (
        "[SKIP] mouse reversal@3 (block 1): reward magnitudes before reversal were [4, 1, 1] "
        "across towers ['A', 'B', 'C'] (expected a permutation of [4, 1, 0])\n"
    )


def test_skipped_neighbour_still_bounds_the_post_window():
    rewards = {"A": [4] * 10, "B": [1] * 10, "C": [0] * 10}
    rewards["C"][6] = 3  # reversal@7 fails the pattern
    reversals = get_bad_reversal_info(one_subject(make_session(BAD, rewards=rewards)))["mouse"]
    assert [r["reversal_idx"] for r in reversals] == [3]
    assert reversals[0]["trial_window_idx"]["post"] == [3, 4, 5, 6]


def test_custom_required_reward_pattern_accepts_other_magnitudes():
    rewards = {"A": [3] * 10, "B": [3] * 10, "C": [0] * 10}
    session = make_session(BAD, rewards=rewards)
    assert get_bad_reversal_info(one_subject(session))["mouse"] == []
    kept = get_bad_reversal_info(one_subject(session), required_reward_pattern=[0, 3, 3])["mouse"]
    assert [r["reversal_idx"] for r in kept] == [3, 7]


def test_short_reward_arrays_are_skipped_before_the_reversal_and_none_after(capsys):
    rewards = {"A": [4] * 4, "B": [1] * 4, "C": [0] * 2}  # C has no row for trial 2
    out = get_bad_reversal_info(one_subject(make_session(BAD, rewards=rewards)))["mouse"]
    assert out == []
    assert (
        "[SKIP] mouse reversal@3: reward_magnitudes_by_tower too short for idx-1"
        in capsys.readouterr().out
    )

    rewards = {"A": [4] * 10, "B": [1] * 10, "C": [0] * 10}
    rewards["A"] = rewards["A"][:5]  # reaches trial 2 and 4 but not trial 7
    out = get_bad_reversal_info(one_subject(make_session(BAD, rewards=rewards)))["mouse"]
    assert [r["reversal_idx"] for r in out] == [3]  # reversal@7 has no reward row before it
    assert out[0]["reward_magnitudes_by_tower_after"] == {"A": 4, "B": 1, "C": 0}


def test_reward_after_is_none_for_every_tower_when_any_array_ends_at_the_reversal():
    rewards = {"A": [4] * 10, "B": [1] * 10, "C": [0] * 10}
    rewards["C"] = rewards["C"][:7]  # trial 6 exists, trial 7 does not
    out = get_bad_reversal_info(one_subject(make_session(BAD, rewards=rewards)))["mouse"]
    last = out[-1]
    assert last["reversal_idx"] == 7
    assert last["reward_magnitudes_by_tower_after"] == {"A": None, "B": None, "C": None}
    assert last["reward_magnitudes_by_tower_before"] == {"A": 4, "B": 1, "C": 0}


def test_choice_slices_drop_indices_past_the_end_of_each_array():
    choices = {"A": [1] * 10, "B": [1] * 5, "C": [1] * 10}
    out = get_bad_reversal_info(one_subject(make_session(BAD, choices=choices)), pre=2, post=3)[
        "mouse"
    ]
    assert out[1]["choices_by_tower"]["B"] == {"pre": [], "post": []}
    assert out[1]["choices_by_tower"]["A"] == {"pre": [1, 1], "post": [1, 1, 1]}


def test_towers_come_from_the_reward_data_even_when_choices_name_more_towers():
    choices = {t: [1] * 10 for t in ("A", "B", "C", "D")}
    out = get_bad_reversal_info(one_subject(make_session(BAD, choices=choices)))["mouse"]
    assert out[0]["towers"] == ["A", "B", "C"]
    assert list(out[0]["choices_by_tower"]) == ["A", "B", "C", "D"]


def test_towers_without_reward_data_are_skipped_with_a_message(capsys):
    session = make_session(BAD)
    session["reward_magnitudes_by_tower"] = {}
    assert get_bad_reversal_info(one_subject(session))["mouse"] == []
    out = capsys.readouterr().out
    assert (
        "[SKIP] mouse reversal@3: missing towers in reward_magnitudes_by_tower: ['A', 'B', 'C']"
        in out
    )
    assert "[SKIP] mouse reversal@7" in out


def test_no_towers_at_all_warns_and_returns_empty(capsys):
    session = make_session(BAD)
    session["reward_magnitudes_by_tower"] = {}
    session["choices_by_tower"] = {}
    assert get_bad_reversal_info(one_subject(session))["mouse"] == []
    assert capsys.readouterr().out == (
        "[WARN] mouse: no towers found in reward_magnitudes_by_tower or choices_by_tower\n"
    )


def test_blocks_length_mismatch_warns_and_block_id_is_none_past_the_end(capsys):
    session = make_session(BAD, blocks=[1, 1, 1, 2, 2])
    out = get_bad_reversal_info(one_subject(session))["mouse"]
    assert capsys.readouterr().out == "[WARN] mouse: blocks length 5 != bad_reversals length 10\n"
    assert [r["block_id"] for r in out] == [2, None]


def test_subject_without_bad_counter_or_trials_gets_an_empty_list():
    no_counter = make_session(BAD)
    del no_counter["bad_reversals"]
    data = {"nocounter": {"ses-1": no_counter}, "empty": {}}
    assert get_bad_reversal_info(data) == {"nocounter": [], "empty": []}
    assert get_bad_reversal_info({}) == {}


def test_sessions_are_merged_so_counters_continue_across_them():
    first = make_session([0, 0, 1, 1])
    second = make_session([0, 1, 1, 1])
    first["reversal_totals"] = {"bad": 1}
    data = {"mouse": {"ses-1": first, "ses-2": second}}
    out = get_bad_reversal_info(data)["mouse"]
    assert [(r["reversal_idx"], r["bad_reversal_number"]) for r in out] == [(2, 1), (5, 2)]


def test_each_subject_is_processed_independently_and_input_is_not_mutated():
    data = {
        "m1": {"ses-1": make_session(BAD)},
        "m2": {"ses-1": make_session([0, 0, 1, 1])},
    }
    before = copy.deepcopy(data)
    out = get_bad_reversal_info(data)
    assert list(out) == ["m1", "m2"]
    assert [r["reversal_idx"] for r in out["m1"]] == [3, 7]
    assert [r["reversal_idx"] for r in out["m2"]] == [2]
    assert data == before
