"""Geometry derived from a block-length summary for plotting (no plotting)."""

from __future__ import annotations

import numpy as np


def tallest_bar(meds: np.ndarray, ses: np.ndarray) -> float:
    """Top of the highest error bar, ``max(median + SE)``; 1.0 when there are no bars."""
    return float(np.nanmax(meds + ses)) if meds.size else 1.0


def annotation_height(meds: np.ndarray, ses: np.ndarray) -> float:
    """Default y for the bar-value labels: 5% above the tallest error bar (at least 1)."""
    return max(tallest_bar(meds, ses), 1.0) * 1.05


def y_upper_limit(annotate_y: float, meds: np.ndarray, ses: np.ndarray) -> float:
    """Upper y-limit that fits both the labels at *annotate_y* and the tallest bar."""
    return max(annotate_y, tallest_bar(meds, ses)) * 1.08


def mouse_points(blocklens: dict, blocks: list) -> tuple[np.ndarray, np.ndarray]:
    """Positions (index into *blocks*) and lengths of the blocks one mouse has data for.

    Args:
        blocklens: ``{block_number: length}`` for one mouse.
        blocks: All block numbers, in plotting order.
    """
    positions, lengths = [], []
    for position, block in enumerate(blocks):
        if block in blocklens:
            positions.append(position)
            lengths.append(blocklens[block])
    return np.asarray(positions, float), np.asarray(lengths, float)
