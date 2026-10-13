"""Malformed values must raise at the source instead of silently changing counts."""

import math
from datetime import datetime

import pytest

from src.behavior_analysis.merge_across_problems import merge_first_leave_dicts, sum_count_dicts
from src.behavior_analysis.problem_utils import (
    mean_block_length_from_indices,
    reversals_from_blocks_counter,
)
from src.behavior_import.extract_trials import add_dicts
from src.behavior_import.import_data import parse_content_json, parse_file_timestamp


# --- parse_content_json -------------------------------------------------------

def test_parse_content_json_object():
    assert parse_content_json('{"trial": 3, "choice": "A"}') == {"trial": 3, "choice": "A"}


def test_parse_content_json_plain_text_rows_become_empty():
    # pyControl event/state/info rows are plain text, not JSON
    assert parse_content_json("poke_A") == {}


def test_parse_content_json_missing_and_preparsed():
    assert parse_content_json(float("nan")) == {}
    assert parse_content_json({"a": 1}) == {"a": 1}


def test_parse_content_json_unexpected_type_raises():
    with pytest.raises(TypeError):
        parse_content_json(object())


# --- count merging ------------------------------------------------------------

def test_sum_count_dicts_sums_valid_counts():
    out = sum_count_dicts([{"new_best": 1, "total": 2}, {"third": 1, "total": 1}, {"total": None}])
    assert out == {"new_best": 1, "third": 1, "total": 3}


def test_sum_count_dicts_rejects_non_numeric_count():
    with pytest.raises(TypeError, match="total"):
        sum_count_dicts([{"total": 1}, {"total": "2"}])


def test_merge_first_leave_dicts_rejects_non_numeric_count():
    with pytest.raises(TypeError, match="new_best"):
        merge_first_leave_dicts({1: {"m1": {"new_best": [1], "third": 0, "total": 1}}})


def test_merge_first_leave_dicts_sums_across_problems():
    merged = merge_first_leave_dicts({
        1: {"m1": {"new_best": 1, "third": 0, "total": 2}},
        2: {"m1": [{"new_best": 2, "third": 1, "total": 3}]},
    })
    assert merged == {"m1": {"new_best": 3, "third": 1, "total": 5}}


def test_add_dicts_sums_counters():
    assert add_dicts({"best": 2, "second": 1}, {"best": 3}) == {"best": 5, "second": 1}


def test_add_dicts_rejects_non_integer_counter():
    with pytest.raises((TypeError, ValueError)):
        add_dicts({"best": 2}, {"best": None})


# --- block helpers ------------------------------------------------------------

def test_mean_block_length_ignores_out_of_range_indices():
    assert mean_block_length_from_indices([10, 20, 99], T=50) == 10.0


def test_mean_block_length_rejects_non_integer_index():
    with pytest.raises((TypeError, ValueError)):
        mean_block_length_from_indices([10, "twenty"], T=50)


def test_reversals_from_blocks_counter_skips_missing_trials():
    assert reversals_from_blocks_counter({"blocks": [1, None, 2, 3]}) == 2.0
    assert math.isnan(reversals_from_blocks_counter({}))


def test_reversals_from_blocks_counter_rejects_garbage():
    with pytest.raises(ValueError):
        reversals_from_blocks_counter({"blocks": [1, "two", 3]})


# --- file timestamps ----------------------------------------------------------

def test_parse_file_timestamp_from_filename(tmp_path):
    p = tmp_path / "MY_72_L-2026-03-30-120049.tsv"
    assert parse_file_timestamp(p) == datetime(2026, 3, 30, 12, 0, 49)


def test_parse_file_timestamp_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_file_timestamp(tmp_path / "no-timestamp.tsv")


def test_merge_first_leave_dicts_rejects_unsupported_structure():
    with pytest.raises(TypeError, match="unsupported"):
        merge_first_leave_dicts({1: {"m1": 3}})


def test_merge_first_leave_dicts_skips_subjects_without_data():
    assert merge_first_leave_dicts({1: {"m1": None, "m2": [[], []]}}) == {}
