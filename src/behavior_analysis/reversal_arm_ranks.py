"""Rank the three arms around a reversal, label them, and type each reversal.

Arms are ranked by reward magnitude on the trial before the reversal (the *prev* ranking) and
on the reversal trial (the *new* ranking). A tie leaves the order of the tied arms undefined,
and nothing here breaks it (not by arm name, not by listing order):

- the labels of a scheme (:func:`arm_labels`) exist only when the ranking the scheme uses has
  no tie;
- the type of a reversal (:func:`classify_reversal_type`) is the type under *every* ranking
  that the ties allow, and ``"ambiguous"`` when those disagree.

:func:`select_reversals` applies both rules and counts what it leaves out.

Reversal types are defined by the prev ranking: ``best_to_third`` means the arm that was best
before is third after, ``second_to_third`` that the arm that was second is third after. The
type is the same whichever labelling scheme is plotted.
"""

from collections.abc import Mapping
from itertools import permutations
from typing import NamedTuple

LABEL_SCHEMES = ("prev", "new")
REVERSAL_TYPES = ("all", "best_to_third", "second_to_third")
SCHEME_ROLES = {
    "prev": ("prev_best", "prev_second", "prev_third"),
    "new": ("new_best", "new_second", "new_third"),
}
AMBIGUOUS = "ambiguous"
OTHER = "other"
_BEFORE = "reward_magnitudes_by_tower_before"
_AFTER = "reward_magnitudes_by_tower_after"


class SelectionCounts(NamedTuple):
    """What :func:`select_reversals` did with the reversals of all subjects.

    ``candidates = tied_labels + tied_type + wrong_type + selected``.

    ``tied_labels`` were left out because a tie leaves the scheme's labels undefined;
    ``tied_type`` because their type depends on a tie (type-specific sets only);
    ``wrong_type`` cannot be of the requested type under any ordering of ties. In a
    type-specific set, ties are counted only for reversals that could be of that type.

    Attributes:
        candidates (int): Reversals considered, over all subjects.
        tied_labels (int): Left out because a tie leaves the scheme's labels undefined.
        tied_type (int): Left out because their type depends on a tie.
        wrong_type (int): Left out because they cannot be of the requested type.
        selected (int): Kept.
    """

    candidates: int
    tied_labels: int
    tied_type: int
    wrong_type: int
    selected: int


def consistent_rankings(magnitudes: Mapping[str, float]) -> list[tuple[str, str, str]]:
    """Every best-to-worst ordering of the arms that the magnitudes allow.

    Args:
        magnitudes (Mapping[str, float]): ``{arm: reward magnitude}`` for exactly three arms.

    Returns:
        list[tuple[str, str, str]]: One ordering if there is no tie; one per way of ordering
        the tied arms otherwise. Each ordering lists the arms best to worst.

    Raises:
        ValueError: If there are not three arms or a magnitude is missing (``None``/NaN).
    """
    if len(magnitudes) != 3:
        raise ValueError(f"need exactly three arms, got {list(magnitudes)}")
    if any(v is None or v != v for v in magnitudes.values()):
        raise ValueError(f"missing reward magnitude in {dict(magnitudes)}")
    return [
        order
        for order in permutations(magnitudes)
        if all(magnitudes[a] >= magnitudes[b] for a, b in zip(order, order[1:], strict=False))
    ]


def _type_under(prev_order: tuple, new_order: tuple) -> str:
    """Type of a reversal for one prev ranking and one new ranking."""
    new_third = new_order[2]
    if new_third == prev_order[0]:
        return "best_to_third"
    if new_third == prev_order[1]:
        return "second_to_third"
    return OTHER


def possible_reversal_types(reversal: dict) -> frozenset[str]:
    """Every type the reversal has under some ordering of tied arms (one type if no tie).

    Args:
        reversal (dict): A reversal dict with ``reward_magnitudes_by_tower_before`` and
            ``reward_magnitudes_by_tower_after`` (``{tower: magnitude}``).

    Returns:
        frozenset[str]: A subset of ``"best_to_third"``, ``"second_to_third"`` and ``"other"``.

    Raises:
        ValueError: As :func:`consistent_rankings`.
    """
    return frozenset(
        _type_under(prev, new)
        for prev in consistent_rankings(reversal[_BEFORE])
        for new in consistent_rankings(reversal[_AFTER])
    )


def classify_reversal_type(reversal: dict) -> str:
    """Type of a reversal, defined by the prev ranking.

    Args:
        reversal (dict): A reversal dict with the before and after magnitudes per tower.

    Returns:
        str: ``"best_to_third"`` (the arm that was best before is third after),
        ``"second_to_third"`` (the arm that was second before is third after), ``"other"``,
        or ``"ambiguous"`` when the answer changes with how ties in the before or after
        magnitudes are ordered.

    Raises:
        ValueError: As :func:`consistent_rankings`.
    """
    types = possible_reversal_types(reversal)
    return next(iter(types)) if len(types) == 1 else AMBIGUOUS


def arm_labels(reversal: dict, scheme: str) -> tuple[str, str, str] | None:
    """Arms in best, second, third order of the ranking the scheme uses.

    Args:
        reversal (dict): A reversal dict with the before and after magnitudes per tower.
        scheme (str): ``"prev"`` (ranking before the reversal) or ``"new"`` (ranking after it).

    Returns:
        tuple[str, str, str] | None: ``(best, second, third)`` arm names, or ``None`` if that
        ranking has a tie.

    Raises:
        ValueError: For an unknown scheme, or as :func:`consistent_rankings`.
    """
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"scheme must be one of {LABEL_SCHEMES}, got {scheme!r}")
    rankings = consistent_rankings(reversal[_BEFORE if scheme == "prev" else _AFTER])
    return rankings[0] if len(rankings) == 1 else None


def _check_selectable(reversal_type: str, reversals: list[dict]) -> None:
    if reversal_type not in REVERSAL_TYPES:
        raise ValueError(f"reversal_type must be one of {REVERSAL_TYPES}, got {reversal_type!r}")
    if any(r.get("first_block", False) for r in reversals):
        raise ValueError("a first block record is not a reversal; build windows without it")


def _verdict(reversal: dict, scheme: str, reversal_type: str) -> str:
    """``"selected"`` or the field of :class:`SelectionCounts` that counts the reversal.

    Ties are only reported for reversals that could be of the requested type: one that cannot
    be (under any ordering of its ties) is simply of another type.
    """
    labels_tied = arm_labels(reversal, scheme) is None
    if reversal_type == "all":
        return "tied_labels" if labels_tied else "selected"
    possible = possible_reversal_types(reversal)
    if reversal_type not in possible:
        return "wrong_type"
    if labels_tied:
        return "tied_labels"
    return "tied_type" if len(possible) > 1 else "selected"


def select_reversals(
    reversal_windows: Mapping[str, list[dict]], scheme: str, reversal_type: str
) -> tuple[dict[str, list[dict]], SelectionCounts]:
    """Keep the reversals that have defined labels for *scheme* and are of *reversal_type*.

    Args:
        reversal_windows (Mapping[str, list[dict]]): ``{subject: [reversal dict]}`` as from
            ``get_good_reversal_info``.
        scheme (str): ``"prev"`` or ``"new"``.
        reversal_type (str): ``"all"``, ``"best_to_third"`` or ``"second_to_third"``.

    Returns:
        tuple[dict[str, list[dict]], SelectionCounts]: ``(selected, counts)``:
        ``{subject: reversals}`` without subjects left empty, and
        the :class:`SelectionCounts` over all subjects.

    Raises:
        ValueError: For an unknown scheme or type, a first-block record, or a record whose
            magnitudes are missing or not for three arms.
    """
    _check_selectable(reversal_type, [r for revs in reversal_windows.values() for r in revs])
    tally = dict.fromkeys(SelectionCounts._fields, 0)
    selected: dict[str, list[dict]] = {}
    for subject, reversals in reversal_windows.items():
        for reversal in reversals:
            verdict = _verdict(reversal, scheme, reversal_type)
            tally[verdict] += 1
            tally["candidates"] += 1
            if verdict == "selected":
                selected.setdefault(subject, []).append(reversal)
    return selected, SelectionCounts(**tally)
