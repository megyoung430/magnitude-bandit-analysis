"""First-leave counting must not depend on the hash seed of the interpreter."""

import json
import os
import subprocess
import sys

CODE = """
import json
from src.behavior_analysis.get_first_leave_after_good_reversals import (
    get_first_leave_after_good_reversals as f,
)
# The best tower is T1 before and after, so T2, T3 and T4 all qualify as "third".
rev = {
    "reward_magnitudes_by_tower_before": {"T1": 4, "T2": 1, "T3": 0, "T4": 0},
    "reward_magnitudes_by_tower_after": {"T1": 4, "T2": 0, "T3": 1, "T4": 0},
    "choices_by_tower": {
        "T1": {"pre": [1], "post": [1, 0]},
        "T2": {"pre": [0], "post": [0, 1]},
        "T3": {"pre": [0], "post": [0, 0]},
        "T4": {"pre": [0], "post": [0, 0]},
    },
}
print(json.dumps(f({"m": [rev]})))
"""


def test_first_leave_third_tower_does_not_depend_on_hash_seed():
    # PYTHONHASHSEED is fixed at interpreter start, so each seed needs its own process.
    results = set()
    for seed in range(8):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        out = subprocess.run(
            [sys.executable, "-c", CODE], env=env, capture_output=True, text=True, check=True
        ).stdout
        results.add(out)
    # T2 is the first non-best tower in session order, and it is the first one left to.
    assert {json.loads(r)["m"]["third"] for r in results} == {1}
    assert len(results) == 1
