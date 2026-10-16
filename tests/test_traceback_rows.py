"""A pyControl error traceback written into the log must not scramble the trial order.

pyControl writes an uncaught task exception as a warning row followed by text rows whose
``time`` column holds text (e.g. ``AttributeError:``). Read naively, the whole ``time`` column
becomes strings and sorting by time orders the trials lexically (trial 11 before trial 2).
"""

import json
import warnings

import pandas as pd
import pytest

from src.behavior_import.extract_trials import recording_trial_rows
from src.behavior_import.import_data import read_tsv_file


def _write_log(path, n_trials=12):
    lines = ["time\ttype\tsubtype\tcontent", "0.000\tinfo\ttask_name\tMeg\\3x3_field_blocked"]
    lines += ["0.030\twarning\t\tUnexpected input received from board: uncaught exception"]
    lines += ["AttributeError: ", ""]
    for t in range(1, n_trials + 1):
        content = json.dumps({"num_t": t, "chosen_tow": "A3" if t % 2 else "C3"})
        lines.append(f"{5.0 * t + 3.5:.3f}\tvariable\tprint\t{content}")
    path.write_text("\n".join(lines) + "\n")


def test_traceback_rows_are_dropped_and_time_is_numeric(tmp_path):
    path = tmp_path / "m-2026-09-23-105303.tsv"
    _write_log(path)
    with pytest.warns(UserWarning, match="non-numeric"):
        df = read_tsv_file(path)
    assert pd.api.types.is_float_dtype(df["time"])
    assert not df["time"].isna().any()
    assert "AttributeError:" not in df["time"].astype(str).str.strip().tolist()


def test_trials_stay_in_print_order_after_a_traceback(tmp_path):
    path = tmp_path / "m-2026-09-23-105303.tsv"
    _write_log(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = read_tsv_file(path)
    rows = recording_trial_rows(df)
    numbers = [content["num_t"] for _, content in rows]
    assert numbers == list(range(1, 13))


def test_a_clean_log_is_read_without_a_warning(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("time\ttype\tsubtype\tcontent\n0.000\tstate\t\twait_init\n1.5\tstate\t\tITI\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = read_tsv_file(path)
    assert df["time"].tolist() == [0.0, 1.5]
