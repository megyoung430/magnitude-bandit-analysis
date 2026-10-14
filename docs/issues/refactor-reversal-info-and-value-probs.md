---
type: infrastructure
complexity: simple
status: in progress
---

# Split get_bad_reversal_info and accumulate_choice_probs

Refactor the long functions below into small, single-purpose functions (principle 4), separating analysis from plotting (principles 2-3), with identical behaviour.

- **Files:** `src/behavior_analysis/get_bad_reversal_info.py`, `src/behavior_analysis/get_choice_probs_by_value.py`
- **Functions over 60 lines:** get_bad_reversal_info (150), accumulate_choice_probs (133)

## Acceptance criteria
- Every function in scope is at most ~60 lines of code; public names, signatures, defaults, return values unchanged.
- An EXACT before/after comparison against main is identical (raw array bytes/dtype/shape, dict key order, value types, printed output, exception type and message, input mutation).
- New behaviour is pinned by tests through the PUBLIC functions with synthetic fixtures; deliberate one-line mutations of the logic make tests fail.
- `uv run pytest -q` passes; ruff check + format --check pass on touched files.
- Pre-existing bugs are documented with a present-tense `Known issue:` comment, not fixed.
