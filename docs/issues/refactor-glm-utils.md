---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Break up GLM input formatting

Refactor the long functions in `src/behavior_analysis/glm_utils.py` to follow the delab coding principles without changing behaviour.

Targets: format_glm_input_by_rank (181 lines), format_glm_input (106).

Pure data shaping: golden-test returned design matrices/frames exactly (dtypes, column order, index) for every option on the sample session and synthetic sessions.

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


## Outcome

`format_glm_input` and `format_glm_input_by_rank` are split into small pure helpers (design labels, lag shifting,
history terms, design matrix, per-subject designs) with keyword-only options. Behaviour is identical to `main`
(d7e85d6), including on malformed input: the same exception type and message, and the same returned bytes, dtypes,
shapes, labels and dict key order. This was checked by running both implementations over 6,540 calls (a real sample
session, multi-session subjects with keys `ses-10`/`ses-2`/`ses-3`, 0/1/2/3-trial sessions, ties, `None`/NaN rewards
on chosen and unchosen towers, reward lists shorter or longer than `trial`, short choice series, 1-5 towers,
`n_lags` 0/1/2/4/10, both `drop_redundant_cols`, both `use_believed`) and comparing raw results exactly.

Behaviour deliberately kept from `main` (documented in the docstrings, not fixed):
- Only subjects with exactly three towers are supported and this is not checked up front: two towers raise
  `KeyError('Arm3')`, four or more raise numpy's "could not broadcast input array" `ValueError`.
- A `None` reward on a chosen tower raises `TypeError` when it enters a lag (`n_lags >= 1`); it is never turned
  into NaN. A missing last-trial reward, or any reward with `n_lags=0`, is ignored.
- A choice series shorter than `trial` raises `IndexError`; a reward list one shorter than `trial` is accepted
  and, in a session that is not last, shifts every later session's rewards one trial earlier after merging.
- Sessions are concatenated before the design is built, so lags (and the believed-rank state of
  `format_glm_input_by_rank`) cross session boundaries.
- `format_glm_input_by_rank` keeps rows whose own outcome is NaN (no single chosen tower, or `n_lags=0` with
  `use_believed`); only incomplete history drops a row.
- `format_glm_input` returns `col_labels` without the `Arm3` columns even when `drop_redundant_cols=False`
  keeps them.
