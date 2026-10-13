---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Split markov_bandit simulations into simulate / summarise / plot

Refactor the long functions in `src/tasks/markov_bandit.py` to follow the delab coding principles without changing behaviour.

Targets: sample_two_markov_bandits (113 lines), simulate_3armed_blockswaps_markov (95), sweep_markov_metrics (69), plot_example_session_markov (66).

Simulations are random: golden-test with fixed seeds / fixed numpy Generator and require identical outputs. Separate plotting (ax=None) from simulation. Do not change the random-number consumption order.

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
