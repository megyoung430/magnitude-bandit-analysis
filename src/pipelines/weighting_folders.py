"""Subfolder names for the two ways a figure can average over its data.

An averaged figure comes in two versions, each in its own subfolder of the analysis folder:
mouse-weighted (each mouse's events pooled into one curve, then mean and SE across mice) and
event-weighted (every event counts once, mean and SE across events).
"""

MOUSE_WEIGHTED_FOLDER = "mouse-weighted"
EVENT_WEIGHTED_FOLDER = "event-weighted"

# The ``weighting`` argument of the analyses (``"subject"`` or ``"event"``) -> subfolder.
WEIGHTING_FOLDERS = {"subject": MOUSE_WEIGHTED_FOLDER, "event": EVENT_WEIGHTED_FOLDER}
