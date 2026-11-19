"""Quality control of the fiber/rotary-joint test recordings of 13 Nov 2026.

Each rotary-joint × patch-cord combination was recorded with the tip in air (baseline), while
paper was touched to the tip, and on two implanted mice. The measures below are chosen so that
combinations can be compared despite different LED currents and a manual paper touch:

- background: the tip-in-air level, which is light generated or reflected in the patch cord and
  joint rather than in the brain. It adds to every recording as a constant offset, dilutes ΔF/F
  and contributes its own bleaching drift.
- brain fraction: (mouse − background) / mouse, the share of recorded light that comes from the
  animal.
- brain signal per mA: (mouse − background) / LED current, a throughput measure independent of
  the LED setting.
- noise after motion correction, expressed relative to the brain-derived light rather than the
  total, since that is the ΔF/F noise floor once background is subtracted properly.
- steps: abrupt shifts in the mouse recordings, the signature of a moving connection.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sps

from pyphotometry import import_ppd

DATA = Path("/mnt/user-data/uploads/Desktop/fiber tests")
KINDS = ("baseline", "paper", "RT_176", "RT_65")
LOWPASS_HZ = 10.0
STEP_WINDOW_S = 0.5
STEP_THRESHOLD_PCT = 0.5  # of the total signal; the smallest step visible by eye in the traces
STEP_MERGE_S = 1.0
TOUCH_FRACTION = 0.5  # of the way from off-level to peak, on the 405 channel


def parse_name(stem: str) -> dict:
    match = re.match(r"rotary(\d)_fiber(\d)_(.*)-(2026-11-13-\d{6})", stem)
    rotary, fiber, rest, stamp = match.groups()
    clean = rest.lower().replace("basline", "baseline")
    kind = next(k for k in KINDS if k.lower() in clean)
    return {"combo": f"R{rotary} F{fiber}", "rotary": int(rotary), "fiber": int(fiber),
            "kind": kind, "label": rest, "time": stamp[-6:]}


def load(path: Path) -> dict:
    data = import_ppd(path, low_pass=None, high_pass=None)
    return {**parse_name(path.stem), "fs": data["sampling_rate"],
            "led_signal_ma": data["LED_current"][0], "led_control_ma": data["LED_current"][1],
            "signal": data["analog_1"], "control": data["analog_2"]}


def lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    return sps.sosfiltfilt(sps.butter(2, cutoff / (fs / 2), output="sos"), x)


def touch_mask(control: np.ndarray) -> np.ndarray:
    """Samples during a paper touch, found on the 405 channel where touches are unmistakable."""
    off = np.percentile(control, 10)
    return control > off + TOUCH_FRACTION * (control.max() - off)


def paper_measures(rec: dict) -> dict:
    """Off-touch level (the tip in air again) and the 465 response to the paper."""
    touching = touch_mask(rec["control"])
    # Leave a margin either side of each touch so the rising and falling edges are excluded.
    margin = int(0.3 * rec["fs"])
    near = np.convolve(touching, np.ones(2 * margin + 1), mode="same") > 0
    off_signal = np.median(rec["signal"][~near])
    off_control = np.median(rec["control"][~near])
    labels, n_touches = _label_runs(touching)
    peaks = [rec["signal"][labels == i].max() - off_signal for i in range(1, n_touches + 1)]
    return {"paper_off_signal_v": off_signal, "paper_off_control_v": off_control,
            "paper_n_touches": n_touches,
            "paper_signal_step_v": float(np.median(peaks)) if peaks else np.nan,
            "paper_control_clipped": bool(rec["control"].max() > 3.25)}


def _label_runs(mask: np.ndarray) -> tuple:
    edges = np.diff(np.concatenate([[0], mask.astype(int)]))
    labels = np.cumsum(edges == 1) * mask
    return labels, int(labels.max())


def baseline_measures(rec: dict) -> dict:
    """Level, bleaching drift and fast noise with the tip in air."""
    out = {}
    minutes = np.arange(len(rec["signal"])) / rec["fs"] / 60
    for channel in ("signal", "control"):
        x = rec[channel]
        slope = np.polyfit(minutes, x, 1)[0]
        out[f"baseline_{channel}_v"] = float(np.median(x))
        out[f"baseline_{channel}_drift_pct_per_min"] = float(100 * slope / np.median(x))
        out[f"baseline_{channel}_noise_pct"] = float(
            100 * np.std(x - lowpass(x, rec["fs"], LOWPASS_HZ)) / np.median(x))
    return out


def count_steps(x: np.ndarray, fs: float) -> tuple:
    """Abrupt shifts: the change across a short window, as a percentage of the level."""
    smooth = lowpass(x, fs, 2.0)
    half = int(STEP_WINDOW_S * fs)
    change = 100 * (smooth[2 * half:] - smooth[:-2 * half]) / np.median(x)
    above = np.abs(change) > STEP_THRESHOLD_PCT
    # Runs closer than STEP_MERGE_S are one step.
    gap = int(STEP_MERGE_S * fs)
    merged = np.convolve(above, np.ones(gap), mode="same") > 0
    _, n_steps = _label_runs(merged)
    return n_steps, float(np.abs(change).max())


def mouse_measures(rec: dict, background: dict) -> dict:
    fs, minutes = rec["fs"], len(rec["signal"]) / rec["fs"] / 60
    out = {"duration_min": minutes}
    detrended = {}
    for channel in ("signal", "control"):
        x = rec[channel]
        level = float(np.median(x))
        brain = level - background[channel]
        out[f"{channel}_v"] = level
        out[f"{channel}_brain_v"] = brain
        out[f"{channel}_brain_fraction"] = brain / level
        out[f"{channel}_brain_v_per_ma"] = brain / rec[f"led_{channel}_ma"]
        steps, largest = count_steps(x, fs)
        out[f"{channel}_steps_per_min"] = steps / minutes
        out[f"{channel}_largest_step_pct"] = largest
        t = np.arange(len(x)) / fs
        filtered = lowpass(x, fs, LOWPASS_HZ)
        detrended[channel] = filtered - np.polyval(np.polyfit(t, filtered, 2), t)

    slope, intercept = np.polyfit(detrended["control"], detrended["signal"], 1)
    residual = detrended["signal"] - (slope * detrended["control"] + intercept)
    total = np.var(detrended["signal"])
    out["motion_r2"] = float(1 - np.var(residual) / total) if total > 0 else np.nan
    # Relative to brain-derived light, i.e. the ΔF/F noise floor once background is removed.
    out["raw_sd_pct_of_brain"] = float(100 * np.std(detrended["signal"]) / out["signal_brain_v"])
    out["corrected_sd_pct_of_brain"] = float(100 * np.std(residual) / out["signal_brain_v"])
    out["corrected_sd_pct_of_total"] = float(100 * np.std(residual) / out["signal_v"])
    return out


def seconds_of_day(stamp: str) -> int:
    return int(stamp[:2]) * 3600 + int(stamp[2:4]) * 60 + int(stamp[4:])


def tip_in_air_points(combo: str, recordings: list, led: tuple) -> list:
    """Every tip-in-air level measured for one combination at one LED setting, in time order.

    Baseline recordings and the off-touch stretches of paper recordings are the same physical
    condition, so both count.
    """
    points = []
    for rec in recordings:
        if rec["combo"] != combo or (rec["led_signal_ma"], rec["led_control_ma"]) != led:
            continue
        if rec["kind"] == "baseline":
            level = {"signal": float(np.median(rec["signal"])),
                     "control": float(np.median(rec["control"]))}
        elif rec["kind"] == "paper":
            paper = paper_measures(rec)
            level = {"signal": paper["paper_off_signal_v"], "control": paper["paper_off_control_v"]}
        else:
            continue
        points.append((seconds_of_day(rec["time"]), level))
    if not points:
        raise ValueError(f"no tip-in-air recording for {combo} at LED {led}.")
    return sorted(points, key=lambda p: p[0])


def background_for(rec: dict, recordings: list) -> dict:
    """Tip-in-air level at the time of a mouse recording.

    The patch cord bleaches by a few percent per minute, so a single tip-in-air value taken
    minutes earlier overstates the background. Where measurements bracket the recording the
    level is interpolated linearly between them; otherwise the nearest one is used and the
    estimate is flagged as one-sided.
    """
    led = (rec["led_signal_ma"], rec["led_control_ma"])
    points = tip_in_air_points(rec["combo"], recordings, led)
    t = seconds_of_day(rec["time"])
    times = np.array([p[0] for p in points])
    out = {}
    for channel in ("signal", "control"):
        levels = np.array([p[1][channel] for p in points])
        out[channel] = float(np.interp(t, times, levels))
    bracketed = times.min() < t < times.max()
    out["source"] = "interpolated" if bracketed else ("after only" if t < times.min()
                                                      else "before only")
    out["spread_pct"] = float(100 * (np.ptp([p[1]["signal"] for p in points]))
                              / np.mean([p[1]["signal"] for p in points]))
    return out


def main() -> None:
    recordings = [load(p) for p in sorted(DATA.glob("*.ppd"))]

    per_recording = []
    for rec in recordings:
        row = {k: rec[k] for k in ("combo", "rotary", "fiber", "kind", "label", "time",
                                   "led_signal_ma", "led_control_ma")}
        if rec["kind"] == "baseline":
            row.update(baseline_measures(rec))
        elif rec["kind"] == "paper":
            row.update(paper_measures(rec))
        else:
            background = background_for(rec, recordings)
            row["background_source"] = background["source"]
            row["background_spread_pct"] = background["spread_pct"]
            row.update(mouse_measures(rec, background))
        per_recording.append(row)

    table = pd.DataFrame(per_recording)
    table.to_csv("fiber_qc_per_recording.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    mice = table[table["kind"].isin(("RT_176", "RT_65"))]
    print(mice[["combo", "label", "led_signal_ma", "signal_v", "signal_brain_v",
                "signal_brain_fraction", "control_brain_fraction", "signal_brain_v_per_ma",
                "motion_r2", "raw_sd_pct_of_brain", "corrected_sd_pct_of_brain",
                "signal_steps_per_min", "signal_largest_step_pct",
                "background_source", "background_spread_pct"]].round(3).to_string(index=False))
    print()
    base = table[table["kind"] == "baseline"]
    print(base[["combo", "label", "led_signal_ma", "baseline_signal_v", "baseline_control_v",
                "baseline_signal_drift_pct_per_min", "baseline_control_drift_pct_per_min",
                "baseline_signal_noise_pct", "baseline_control_noise_pct"]].round(4).to_string(index=False))
    print()
    paper = table[table["kind"] == "paper"]
    print(paper[["combo", "led_signal_ma", "paper_n_touches", "paper_off_signal_v",
                 "paper_signal_step_v", "paper_control_clipped"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
