---
type: infrastructure
complexity: complex
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Split choice-probabilities-around-reversals analysis

Refactor the long functions in `src/behavior_analysis/get_choice_probs_around_good_reversals.py` to follow the delab coding principles without changing behaviour.

Targets: get_choice_probs_around_good_reversals (142 lines), apply_moving_average_to_choice_probs (136), remove_trials_after_bad_rev (95).

Golden-test every keyword combination the pipeline uses (skip_n_trials_after_reversal, moving-average modes 'centered' and any others the code supports, include_bad_trial True/False) on windows from get_good_reversal_info on the sample session and on synthetic windows.

## Method (applies to every refactor item)
Behaviour-preserving refactor, test-first (principle 7):
1. BEFORE touching code, write a capture script in `.golden/` inside your worktree (git-ignored, not committed) using
   /home/claude/golden/goldkit.py (import read-only; `canon`, `capture_figures`, `save`, `compare`). Record return values of every
   target function, and for plotting functions every figure drawn/saved (capture_figures), over inputs that cover
   every branch: synthetic inputs you construct, plus the sample session in
   data/3x3_field_blocked_reward_bandit/cohort-01/rawdata (load with
   `src.behavior_import.load_trials.load_trials_by_problem(root, use_cache=False)`). Save before.json.
2. Refactor. After: the same script must report IDENTICAL via compare().
3. Add pytest unit tests with small synthetic fixtures for each new pure helper (tests/test_<module>.py).

## Acceptance criteria
- Every target function is <= 60 lines (aim for <= 40); helpers are small, named for what they do (principle 4).
- Computation is separated from plotting; panel-drawing functions accept `ax=None` and create a figure only if
  none is given (principle 3). Saving/showing stays in the public function.
- Public function names, signatures, defaults, return values and saved file names are unchanged, so scripts and
  notebooks keep working. Do not edit notebooks or other modules except for import lines.
- Golden capture reports IDENTICAL; include the case count in the report.
- Prefer comprehensions/vectorised operations over index loops where it doesn't change results (principle 5);
  keep explicit loops where state is sequential.
- No new broad `except`; no silent fallbacks (principle 9). Comments explain the present (principle 10).
- If you find a bug, DO NOT fix it: preserve behaviour, add a short comment stating the issue, and report it.
- `uv run --no-sync ruff check <touched and new files>` clean; `uv run --no-sync ruff format` only touched/new files.
- `uv run --no-sync pytest -q` passes in your worktree.
- Commit on your branch in your worktree only, small commits, messages per principle 10. Set
  `status: in review` in this file in your final commit.
