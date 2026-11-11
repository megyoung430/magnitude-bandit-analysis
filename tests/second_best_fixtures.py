"""A hand-built mouse for the second-best tests, with every switch worked out by hand.

Ports sort as A1 < B3 < C2 (arm 0, 1, 2). Each row is ``(problem, session, trial in session,
choice, magnitudes (A1, B3, C2))``; the reward is the chosen arm's magnitude, as in the task.

Switch trials (``*`` = leaves the strict best of the previous trial, ``t`` = top tied):

====  =====  =============  ===========  ==========  ===========================
row   from   previous mags  alternatives  condition   notes
====  =====  =============  ===========  ==========  ===========================
r1    A      4 2 0          B, C          non-zero    *, chooses the second best
r2    B      4 2 0          A, C          non-zero    B was not the best
r3    C      3 3 0          A, B          zero        true tie A=B at t
r5    A      3 3 0          B, C          non-zero    t: top tied (A and B)
r6    B      2 3 1          A, C          non-zero    *, chooses the second best
r7    A      2 3 1          B, C          non-zero    A was not the best
r8    C      2 3 1          A, B          non-zero    C was not the best
r9    B      2 3 1          A, C          non-zero    *, chooses the third
r11   A      2 3 1          B, C          non-zero    second session of problem 1
r13   B      1 4 2          A, C          non-zero    * (problem 2), chooses the third
r14   A      1 4 2          B, C          non-zero
r15   C      1 4 2          A, B          non-zero
r16   B      2 4 2          A, C          non-zero    *, the alternatives tie (A=C)
====  =====  =============  ===========  ==========  ===========================

r0, r10 and r12 start a session (no previous choice); r4 stays.
"""

import pandas as pd

from tests.switch_regression_fixtures import table_from_rows

PORTS = ("A1", "B3", "C2")

# (problem, session, trial in session, choice index, magnitudes of A1, B3, C2)
ROWS = (
    (1, 0, 0, 0, (4, 2, 0)),  # r0 start
    (1, 0, 1, 1, (4, 2, 0)),  # r1
    (1, 0, 2, 2, (3, 3, 0)),  # r2
    (1, 0, 3, 0, (3, 3, 0)),  # r3
    (1, 0, 4, 0, (3, 3, 0)),  # r4 stay
    (1, 0, 5, 1, (2, 3, 1)),  # r5
    (1, 0, 6, 0, (2, 3, 1)),  # r6
    (1, 0, 7, 2, (2, 3, 1)),  # r7
    (1, 0, 8, 1, (2, 3, 1)),  # r8
    (1, 0, 9, 2, (2, 3, 1)),  # r9
    (1, 1, 0, 0, (2, 3, 1)),  # r10 start of the second session
    (1, 1, 1, 1, (2, 3, 1)),  # r11
    (2, 0, 0, 1, (1, 4, 2)),  # r12 start of problem 2
    (2, 0, 1, 0, (1, 4, 2)),  # r13
    (2, 0, 2, 2, (1, 4, 2)),  # r14
    (2, 0, 3, 1, (2, 4, 2)),  # r15
    (2, 0, 4, 0, (2, 4, 2)),  # r16
)


def hand_mouse(mouse: str = "m1", magnitudes: dict | None = None) -> pd.DataFrame:
    """The trial table of the hand-built mouse.

    Args:
        mouse (str): Mouse id.
        magnitudes (dict | None): ``{row index: (A1, B3, C2)}`` replacing the magnitudes of those
            rows; the reward follows the chosen arm.
    """
    replaced = magnitudes or {}
    rows = []
    for i, (problem, session, in_session, choice, mags) in enumerate(ROWS):
        mags = replaced.get(i, mags)
        rows.append(
            (problem, session, in_session, False, PORTS[choice], mags[choice],
             dict(zip(PORTS, mags, strict=True)))
        )  # fmt: skip
    return table_from_rows(rows, mouse)
