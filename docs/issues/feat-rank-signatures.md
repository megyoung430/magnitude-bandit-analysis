# feat-rank-signatures

**Results change: the cohort 1 selection.** New analyses, plus one change to what the existing
cohort 1 analyses are run on (see "Results change: the cohort 1 selection"). Nothing that existed
produces different numbers unless it is run on the selected table.

## Request
Test a ranked value algorithm (rank the three options; choices gated by comparisons between
adjacent ranks) against a softmax value learner with five behavioral signatures, each calibrated
against replays of agents fitted per mouse on that mouse's own schedules: WSLS (Magnitude), WSLS
(Habit) and the kernel + forgetting Q-learner (the key comparator). Select the data first, and
check that the earlier baseline numbers do not move.

## Definitions
- **Believed value**: the last payout at an arm in the current problem, carried across sessions.
  It is undefined before the first visit; a state is defined only when all three arms were
  visited, and the others are excluded (the number is reported).
- **Ties** use average ranks, with piecewise-linear first, second and third indicators.
- **Decision**: consecutive trials within one segment (a segment starts at a problem start, a
  session start or after a gap). `own` is the arm chosen, `next` the next choice.
- **Scale**: the magnitude sets are `{0, a, b}` with `a`, `b` in 1..4, so a rank occurs at several
  magnitudes (this is what test 4b uses).
- **Block structure**: exactly one arm pays 0 on every trial. At a block change the task zeroes
  the non-zero arm the mouse chose most and gives its value to the old zero arm (1022 of 1025
  changes in cohorts 1 and 3 after the selection; the zeroed arm was the most chosen in 953).
  So the new best can be inferred without sampling it.
- **Calibration**: z = (observed - replay mean) / replay sd over 100 replays of the agent fitted to
  that mouse; two-sided empirical p = (1 + #extreme) / (n + 1). A mouse passes when p < 0.05 in the
  predicted direction. Across mice: mean ± se of z and the Wilcoxon p (from 6 mice; below that,
  dots only).

## What each test predicts
| Test | Readout | Rank algorithm | Softmax value learner |
|---|---|---|---|
| 1. Stay depends on the second best | `w_best`, `w_worst` of the alternatives' believed values in the stay logit; `w_diff` = worst minus best | `w_best` negative, `w_worst` about 0, `w_diff` large positive | both negative (worst less), small `w_diff` |
| 2. Abrupt switching at crossings | step weight at `D` < 0 (`D` = own value minus best alternative) and its gain over a smooth curve; event alignment to believed and true crossings (`cross_leave_1`, `cross_adopt_1`, `cross_t50`) | P(leave) is a step at `D` = 0; the new best is adopted at once | graded in `D`; adoption follows value differences |
| 3. The third option is categorical | slope of P(worst) over the gap to the second best; worst arm's weight in the top-two choice (IIA check) | flat slope, no weight in the top two | falls with the gap |
| 4. Ranks over values | held-out nats of rank over value in the stay and the switch models; scale slope (stay at the best-ranked own arm against the payout's magnitude) | rank predicts better; scale slope 0 | value predicts better; positive scale slope |
| 5. Transfer and inference | slope of P(best) (first 100 trials) and of trials to threshold across problems; share of first leaves after a block change that go to the old zero arm | improves across problems; goes to the old zero arm | no transfer; the leave goes to the third arm, whose last payout is higher |

Each is run on toy agents first (`tests/rank_fixtures.py`, `src/agents/rank_agent.py`): a rank agent, a
softmax learner with learning rate 1 and a block-inferring rank agent, with the predicted signs
asserted (`tests/test_rank_*.py`). The transfer slope has hand-built learning mice in the unit tests.

## Data selection
`MAX_PROBLEM_BY_COHORT = {1: 7, 3: None}` in the notebook (`problem_selection.select_problems`: one
documented parameter, an entry per cohort, raises on a missing cohort or on a mouse left empty;
`dropped_table` prints what is dropped).
- **Cohort 1: drifting problems 1 to 7.** After a four-week surgery / miniscope break only MY_71_L,
  MY_71_R and MY_72_L continue (problem 8 onward, from 2026-04-28), with degraded behavior;
  MY_72_N and MY_72_R end at problem 7. Meg's "problem 11" counts the four blocked problems before the
  drifting ones: it is drifting problem 7 here. Changing the cutoff to 11 is one edit.
- **Cohort 3: all drifting problems.** The trial table already holds drifting problems only (the
  loader numbers them within the cohort and leaves the blocked ones out), for both cohorts.
- Dropped: 10,351 of 49,572 trials (drifting problems 8 and later of MY_71_L, MY_71_R and MY_72_L).
  14 mice remain, none left empty.
- Every model is refitted on the selected table: the cache keys contain the mouse's trials, so a mouse
  whose data changed is never served a stale fit (`pipelines/rank_baseline.py` reads a mouse from the
  shared strategy cache only when its trials are identical).

## Results change: the cohort 1 selection
Numbers on cohorts 1 and 3, before (all drifting problems) and after the selection (14 mice both).

| | before | after |
|---|---|---|
| held-out nats per trial: WSLS (Magnitude) | 0.7475 | 0.7449 |
| WSLS (Habit) | 0.6684 | 0.6661 |
| Q-learner | 0.7554 | 0.7541 |
| kernel Q-learner | 0.6628 | 0.6614 |
| paired: kernel Q minus WSLS (Magnitude) | -0.0847 | -0.0835 |
| kernel Q minus Q-learner | -0.0926 | -0.0927 |
| WSLS (Habit) minus WSLS (Magnitude) | -0.0791 | -0.0788 |
| WSLS (Habit) minus kernel Q | +0.0056 | +0.0046 |
| switch regression, believed value, after non-zero payout (mean beta; Wilcoxon p) | 0.087 (0.0009) | 0.094 (0.0009) |
| same, after a zero payout | 0.088 (0.013) | 0.089 (0.011) |
| true value, after non-zero / after zero | 0.048 / 0.016 | 0.053 / 0.026 |
| calibrated z, believed after non-zero: WSLS (Magnitude) / (Habit) | 2.20 / 2.30 | 2.17 / 2.28 |

The paired kernel-Q and WSLS (Habit) gaps are unchanged by more than 0.002 nats and the switch
regression's finding (switching follows believed values, more than any of the null agents predict) is
unchanged: the selection does not change the conclusions of the earlier analyses. Anyone citing
cohort-1 numbers from before this change should rerun on the selected table.

## Change
- `src/behavior_analysis/problem_selection.py`: the selection and the table of dropped problems.
- `src/pipelines/rank_baseline.py`: held-out nats of four models and paired comparisons on the
  selected table (shared strategy cache first, else `cross_validate` on the subset).
- `src/behavior_analysis/rank_state.py` (believed values, defined states, average ranks),
  `rank_decisions.py` (the decision table, the kernel Q-learner's values), `rank_logit.py` (ridge
  logistic with Laplace errors and leave-one-problem-out likelihood), `rank_stay.py` (tests 1, 4),
  `rank_crossings.py` (test 2), `rank_third.py` (test 3), `rank_transfer.py` (test 5),
  `rank_switch.py` (rank against value in the switch regression), `rank_signatures.py`
  (`analyze_mouse`, `TESTS`, `PREDICTED`; raises `UnanalyzableError` below 200 decisions),
  `rank_calibration.py` (z, p, passes, summaries), `rank_result.py`.
- `src/agents/rank_agent.py`: the toy rank agent (for tests and the notebook's synthetic section).
- `src/pipelines/rank_signatures.py`: mice, replays through `sim.simulate_policy_mouse` (replays that
  are too short are counted as failed, never silently dropped), calibration, joblib caching keyed on
  the data and a hash of the code.
- `src/behavior_visualization/plot_rank_signatures.py`: seven figures, each saved as pdf, png, svg and
  `.caption.txt` (Stay Comparison Weights, Rank Crossing Step, Rank Crossing Aligned, Third Option,
  Rank vs Value, Transfer Across Problems, Block Change Inference). The y-axis of a dot panel is fitted
  to the dots, because an agent's replay bar can be far wider than the mice differ.
- `scripts/rank_signatures.ipynb` (outputs cleared): parameters (`COHORTS`, `MAX_PROBLEM_BY_COHORT`,
  `OUTPUT_DIR = results/figures/rank-tests`), simulated agents first, the selection, the baseline check,
  the tests on the mice, and a "Reading the figures" cell per test.

## Results on cohorts 1 and 3 (14 mice, replays 100 per mouse and agent)
Against the kernel Q-learner fitted to each mouse (z, Wilcoxon p across mice; the mice passing alone):

| Test | Mice | Kernel Q-learner | z, p | Pass | Verdict |
|---|---|---|---|---|---|
| 1 `w_diff` | +0.199 | +0.048 | +1.19, 0.003 | 3 of 14 | rank-like (modest) |
| 2 `step_weight` | +0.161 | +0.177 | -0.08, 1.0 | 0 of 14 | softmax-like (graded; `cross_t50` censored, not testable) |
| 3 `worst_slope` | -0.065 | -0.149 | +0.70, 0.078 | 1 of 14 | not distinguishable (leans rank) |
| 4 `scale_slope_first` | 0.146 | 0.385 | -2.51, < 0.001 | 10 of 14 | rank-like on scale; held-out nats tie |
| 5 `slope_p100` / `block_inherit` | +0.010 / 0.560 | +0.005 / 0.514 | +0.04 (0.95) / +0.56 (0.24) | 1 / 2 of 14 | softmax-like |

## Limitations
- The toy rank and softmax agents show the readouts separate the two accounts when they differ
  sharply; the mice are between them, so the calibration against the fitted agents carries the result.
- `cross_t50` is censored at 11 for every mouse (the median mouse has not adopted the new best after
  10 trials): no z. MY_66_LL has two problems: no transfer slope.
- Five mice cannot reach p < 0.05 in a two-sided exact Wilcoxon test: the across-mice test needs 6.
