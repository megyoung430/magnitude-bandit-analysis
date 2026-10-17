# Function for opening pyPhotometry data files in Python.
# Copyright (c) Thomas Akam 2018-2025. Licensed under GNU GPL v3.
# Official importer retrieved 2026-10-01:
# https://github.com/pyPhotometry/code/blob/master/tools/data_import.py
# `import_ppd` is the official code unchanged apart from formatting; `read_ppd_header` is
# added alongside it.

"""Read pyPhotometry ``.ppd`` files: the full recording, or just the header."""

import json
import os

import numpy as np
from packaging.version import parse as parse_version
from scipy.signal import butter, filtfilt


def import_ppd(file_path, low_pass=20, high_pass=0.01):
    """Function to import pyPhotometry binary data files into Python. The high_pass
    and low_pass arguments determine the frequency in Hz of highpass and lowpass
    filtering applied to the filtered analog signals. To disable highpass or lowpass
    filtering set the respective argument to None.  Returns a dictionary with the
    following items:
        'filename'      - Data filename
        'subject_ID'    - Subject ID
        'date_time'     - Recording start date and time (ISO 8601 format string)
        'end_time'      - Recording end date and time (ISO 8601 format string)
        'mode'          - Acquisition mode
        'sampling_rate' - Sampling rate (Hz)
        'LED_current'   - Current for LEDs 1 and 2 (mA)
        'version'       - Version number of pyPhotometry
        For each analog signal (x in [1, n_analog_signals]):
            'analog_x'            - Raw analog signal (volts)
            'analog_x_filt'       - Filtered analog signal (volts)
            In pulsed acqusition modes with pyPhotometry version >= 1.1:
            'analog_x_raw_LED_on'   - Analog signal before baseline subtraction (volts)
            'analog_x_raw_baseline' - Baseline signal with LED off (volts).
            If signal clipping can be evaluated (not pulsed mode with version < 1.1):
            'analog_x_clipping' - Samples where analog signal was clipping (bool)
        For each digital signal (y in [1, n_digital_signals]):
            'digital_y'     - Digital signal
            'pulse_inds_y'  - Locations of rising edges on digital signal (samples).
            'pulse_times_y' - Times of rising edges on digital signal (ms).
    """

    # Read data from file --------------------------------------------------------------
    with open(file_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype("<u2"))

    # Extract header information -------------------------------------------------------
    header_dict = json.loads(data_header)
    sampling_rate = header_dict["sampling_rate"]
    volts_per_division = header_dict["volts_per_division"][0]
    acquisition_mode = header_dict["mode"]
    version = parse_version(header_dict["version"])

    # Get version specfic info ---------------------------------------------------------

    if version < parse_version("1.0"):
        n_analog_signals = 2
        n_digital_signals = 2
        pulsed_mode = "time div" in acquisition_mode
    else:
        n_analog_signals = header_dict["n_analog_signals"]
        n_digital_signals = header_dict["n_digital_signals"]
        pulsed_mode = "pulsed" in acquisition_mode

    if version >= parse_version("1.1"):
        has_baselines = pulsed_mode
        ADC_max_value = header_dict["ADC_max_value"]
    else:
        has_baselines = False
        ADC_max_value = 1 << 15

    # Extract signals ------------------------------------------------------------------
    analog = data >> 1  # Analog signal is most significant 15 bits.
    digital = ((data & 1) == 1).astype(int)  # Digital signal is least significant bit.
    clip_threshold = 0.98 * ADC_max_value * volts_per_division
    if has_baselines:  # Raw LED-on and LED-off (baseline) samples saved seperately.
        LED_on_sigs = [
            analog[2 * a :: 2 * n_analog_signals] * volts_per_division
            for a in range(n_analog_signals)
        ]
        baselines = [
            analog[2 * a + 1 :: 2 * n_analog_signals] * volts_per_division
            for a in range(n_analog_signals)
        ]
        # Compute baseline subtracted signals by subtracting baseline from LED-on signal.
        analog_sigs = [
            LED_on_sig - baseline for LED_on_sig, baseline in zip(LED_on_sigs, baselines)
        ]
        # Identify any samples where signal is clipping
        sigs_clipping = [
            np.maximum(LED_on_sig, baseline) > clip_threshold
            for LED_on_sig, baseline in zip(LED_on_sigs, baselines)
        ]
        digital_sigs = [digital[2 * d :: 2 * n_analog_signals] for d in range(n_digital_signals)]
    else:  # Any baseline subtraction was done before saving signals.
        analog_sigs = [
            analog[a::n_analog_signals] * volts_per_division for a in range(n_analog_signals)
        ]
        digital_sigs = [digital[d::n_analog_signals] for d in range(n_digital_signals)]
        sigs_clipping = (
            [analog_sig > clip_threshold for analog_sig in analog_sigs] if not pulsed_mode else None
        )

    # Compute sample times relative to start of recording (ms) -------------------------
    time = np.arange(analog_sigs[0].shape[0]) * 1000 / sampling_rate

    # Filter signals with specified high and low pass frequencies (Hz) -----------------
    if low_pass and high_pass:
        b, a = butter(2, np.array([high_pass, low_pass]) / (0.5 * sampling_rate), "bandpass")
    elif low_pass:
        b, a = butter(2, low_pass / (0.5 * sampling_rate), "low")
    elif high_pass:
        b, a = butter(2, high_pass / (0.5 * sampling_rate), "high")
    if low_pass or high_pass:
        analogs_filt = [filtfilt(b, a, analog_sig) for analog_sig in analog_sigs]
    else:
        analogs_filt = [None] * len(analog_sigs)

    # Extract rising edges for digital inputs ------------------------------------------
    pulse_inds = [1 + np.where(np.diff(digital_sig) == 1)[0] for digital_sig in digital_sigs]
    pulse_times = [pulse_ind * 1000 / sampling_rate for pulse_ind in pulse_inds]

    # Return signals + header information as a dictionary ------------------------------
    data_dict = {
        "filename": os.path.basename(file_path),
        "time": time,
    }
    for a in range(n_analog_signals):
        data_dict[f"analog_{a + 1}"] = analog_sigs[a]
        data_dict[f"analog_{a + 1}_filt"] = analogs_filt[a]
        if has_baselines:
            data_dict[f"analog_{a + 1}_raw_LED_on"] = LED_on_sigs[a]
            data_dict[f"analog_{a + 1}_raw_baseline"] = baselines[a]
        if sigs_clipping:
            data_dict[f"analog_{a + 1}_clipping"] = sigs_clipping[a]
    for d in range(n_digital_signals):
        data_dict[f"digital_{d + 1}"] = digital_sigs[d]
        data_dict[f"pulse_inds_{d + 1}"] = pulse_inds[d]
        data_dict[f"pulse_times_{d + 1}"] = pulse_times[d]
    data_dict.update(header_dict)
    return data_dict


def read_ppd_header(path) -> dict:
    """Read only the JSON header of a ``.ppd`` file, without loading its samples.

    Args:
        path (str | Path): The ``.ppd`` file.

    Returns:
        dict: The header, e.g. ``subject_ID``, ``sampling_rate``, ``mode``, ``LED_current``,
        ``version``.

    Raises:
        ValueError: If the file is empty or its header is not valid JSON.
    """
    with open(path, "rb") as f:
        size_bytes = f.read(2)
        if len(size_bytes) < 2:
            raise ValueError(f"{path} is not a pyPhotometry .ppd file: it has no header.")
        header_bytes = f.read(int.from_bytes(size_bytes, "little"))
    try:
        return json.loads(header_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} is not a pyPhotometry .ppd file: bad header ({error})."
        ) from error
