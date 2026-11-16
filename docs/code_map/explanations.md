# Plain-language explanations for the code map

These show as "In plain words" dropdowns on the code map page (`scripts/build_code_map.py`). They
are for readers new to the methods: what question a file answers, what the method is, and how to
read what a function returns. Each entry starts with a `## <key>` heading: a file path for a
primer on the whole file, or `path::name` for one function. Write for someone who knows the task
but not the statistics; define each term the first time it is used. Keep entries short and
accurate, and update an entry when the code it describes changes.

A few words used throughout:

- **Problem**: one set of three offered towers that a mouse works on for many sessions. Blocked
  problems come first in training; drifting problems follow.
- **Magnitude**: the number of reward pulses (0–4) an arm pays when chosen.
- **Block change**: the moment the task changes which arm pays nothing. In the blocked task a
  *good reversal* changes which tower is best, and a *bad reversal* changes the payouts without
  changing the best tower.
- **Believed value**: the magnitude an arm paid the last time the mouse chose it, a simple guess
  at what the mouse "thinks" the arm is worth.
- **Logit / log-odds**: a probability p re-expressed as log(p / (1 − p)), so that effects add up.
  A weight in a logistic model says how much a feature moves the log-odds of a choice.
- **Held-out**: scored on data the model was not fitted to, which is the honest test of a model.
- **Nats per choice**: the average negative log probability a model gave to the choices the mouse
  actually made. Lower is better; guessing at random among three arms gives log(3) ≈ 1.10.

## src/behavior_analysis/reversals/average_choice_probs.py

Around each good reversal, how often does the mouse pick each tower in the trials before and after
it? This file takes those choice curves, computed per problem, and averages them across problems so
each problem counts equally. That stops a problem with many reversals from dominating the average.
It only computes numbers; the plotting lives in `plot_averaged_choice_probs`.

## src/behavior_analysis/reversals/average_choice_probs.py::average_across_problem_curves

Takes one reversal-aligned choice curve per problem and averages them, giving each problem the same
weight. The result is a single curve (mean and standard error at each trial offset from the
reversal) that summarizes how mice respond to reversals across their whole training.

## src/behavior_analysis/reversals/best_change.py

A check that a recorded block change really shows up in the payouts. The rig logs block changes,
but a log entry is only trustworthy if the magnitudes changed in the expected way. In the blocked
task, a good reversal should change which tower is best. In the drifting task, the arm the mouse
chose most drops to zero at a block change, so the test is simply that the zero arm moved. Changes
that fail this check are flagged as *suspect*; ones that cannot be checked (a magnitude is missing)
are flagged as *uncheckable*.

## src/behavior_analysis/task_statistics/blocklen_distributions.py

Gathers how long each block lasted (in trials), per mouse and problem, for the block-length
histograms. Block length matters because it limits how much a mouse can learn before the payouts
change again.

## src/behavior_analysis/choice_logit/choice_kernel_timescale.py

Mice tend to repeat recent choices regardless of reward; this "choice kernel" or habit can be
measured as weights on the choices 1, 2, 3... trials back. If the habit fades steadily, those
weights shrink by the same factor each trial back (a geometric decay). This file fits that decay
factor to each mouse's weights and turns it into a timescale in trials, so it can be compared with
the habit rate fitted inside the kernel Q-learner. A decay near 1 means a flat kernel: choices
many trials back still matter almost as much as the last one.

## src/behavior_analysis/choice_logit/choice_kernel_timescale.py::fit_decay

For each mouse, fits weight = amplitude × decay^(lag − 1) to the choice-lag weights of the choice
logit. Returns the amplitude, the decay (0–1; how much of the habit survives each extra trial) and
the timescale tau = −1/ln(decay) in trials. A mouse with no clean decaying pattern gets NaN rather
than a misleading number.

## src/behavior_analysis/choice_logit/choice_logit.py

The main descriptive model of choice. On every trial the mouse picks one of three ports. The model
gives each port a score (a "utility") made of three kinds of evidence, and turns the scores into
choice probabilities with a softmax (higher score, higher probability; the three always sum to 1):

- **Reward history**: how many pulses the port paid on each of the last few trials it was chosen
  (and, optionally, a slowly fading "tail" of older rewards).
- **Choice history**: whether the port was chosen 1, 2, 3... trials ago, regardless of reward
  (habit or perseveration).
- **Tower bias**: a fixed preference for some towers over others (for example, a port near the
  start).

The fitted weights say how much each kind of evidence drives choice. The reward part of the score
is also used as the model's estimate of each port's value (the "model values") in later analyses.
The fit uses a small penalty (ridge) that keeps weights from growing wild when data are thin.

## src/behavior_analysis/choice_logit/choice_logit.py::design_matrix

Builds the table the choice logit is fitted to: for every trial and every offered port, the
reward-history, choice-history and tower features described in the file primer, plus which port
the mouse actually chose. Think of it as converting the raw trial log into the columns a regression
needs.

## src/behavior_analysis/choice_logit/choice_logit.py::fit_logit

Finds the weights that make the mouse's actual choices as probable as possible under the model
(maximum likelihood), with a small ridge penalty. Because this problem has a single best answer (it
is convex), one run from zero is enough.

## src/behavior_analysis/choice_logit/choice_logit.py::model_values

For each trial and port, the reward part of the fitted score: how valuable the model thinks each
port is, given what it paid recently. These values are in logit units (bigger means more likely to
be chosen) and feed the value psychometrics and stay/switch analyses.

## src/behavior_analysis/choice_logit/choice_logit.py::weights_table

Lays the fitted weights out as a tidy table, one row per weight, labeled by feature group (reward,
choice, tower) and lag, which is the format the weight plots read.

## src/behavior_analysis/choice_logit/choice_logit_cv.py

How good is the choice logit, really? A model can fit the trials it learned from very well and
still fail on new data. This file fits the model on all but one of a mouse's problems and tests it
on the problem left out, repeating for each problem (leave-one-problem-out cross-validation). It
compares three nested models: reward history only, plus choice history, plus tower biases. The
model values used later also come from these held-out fits, so a trial's value never comes from
weights that were fitted to that very trial.

## src/behavior_analysis/choice_logit/choice_logit_cv.py::cross_validate_mouse

For one mouse, runs the leave-one-problem-out loop: fit on the other problems, score on this one.
Returns each model's held-out score and the held-out model values of the full model.

## src/behavior_analysis/choice_logit/choice_logit_cv.py::model_comparison

Turns the held-out scores into nats per choice per mouse and model (lower is better; random
guessing is about 1.10). Comparing rows tells you, for example, whether adding choice history
really improves prediction on unseen problems.

## src/behavior_analysis/choice_logit/choice_logit_extensions.py

Four follow-up questions asked of the choice logit, each answered by fitting variants of the model
and comparing their held-out scores:

- **Believed value**: does knowing what each arm paid last time (even if that was long ago) help
  beyond recent reward history?
- **Binary vs pulses**: do mice respond to whether they got anything at all, or to how many pulses
  they got?
- **Variance partition**: how much of the predictable behavior is due only to reward history, only
  to choice history, only to tower bias, and how much do reward and choice share?
- **Calibration**: when the model says "70%", does the mouse choose that port about 70% of the
  time?

## src/behavior_analysis/choice_logit/choice_logit_extensions.py::cross_validate_variants

The shared engine for the four questions: fits each requested model variant with
leave-one-problem-out cross-validation and returns their held-out scores and predicted
probabilities.

## src/behavior_analysis/choice_logit/choice_logit_extensions.py::believed_value_table

Per mouse, the weight on believed value (how much the log-odds of choosing an arm rise per pulse it
paid last time), the weight on arms not yet visited, and how much adding believed value improves
held-out prediction. A positive gain means the mouse carries a memory of each arm's last payout
beyond the last few trials.

## src/behavior_analysis/choice_logit/choice_logit_extensions.py::binary_pulse_table

Per mouse, compares three ways of coding reward in the model: as pulse counts, as rewarded or not
(binary), and both. Lower held-out nats wins. If binary wins, the mouse mostly cares whether it got
any reward; if pulses win, it is sensitive to magnitude.

## src/behavior_analysis/choice_logit/choice_logit_extensions.py::gain_summary

Summarizes per-mouse improvements across mice: mean, standard error, how many mice improved, and
two paired tests (Wilcoxon signed-rank and sign test) of whether the improvement is reliably above
zero. Mice are the replicates.

## src/behavior_analysis/choice_logit/choice_logit_extensions.py::variance_partition

Splits what the model explains into parts by leaving groups out. "Unique reward" is how much worse
the model gets without reward history, given the other groups; likewise for choice and tower.
"Shared" is the part reward and choice history explain together but neither alone, which happens
when the two are correlated (mice tend to repeat rewarded choices).

## src/behavior_analysis/choice_logit/choice_logit_extensions.py::calibration_table

Groups trials by the probability the model predicted for a port and, within each group, measures
how often the mouse actually chose it. Points on the identity line (predicted = observed) mean the
model's probabilities can be taken at face value.

## src/behavior_analysis/choice_logit/choice_logit_extensions.py::calibration_noise_floor

Even a perfect model would not land exactly on the identity line, because each bin holds a finite
number of choices. This simulates that: it draws choices from the model's own probabilities and
measures the calibration error that sampling noise alone produces. A real error close to this floor
means the model is as well calibrated as the data can show.

## src/behavior_analysis/choice_logit/choice_regression.py

A simpler choice model (a conditional logit, like the choice logit) using a handful of
hand-picked features for each port, used by the agent-baseline pipeline: the value it paid last time, whether that was zero, whether it
is still unvisited, and whether it was the previous choice. One set of weights is fitted per mouse,
and the mice are the replicates for group statistics.

## src/behavior_analysis/choice_logit/choice_regression.py::fit_by_mouse

Fits the conditional logit to each mouse separately and returns each weight with its standard error
(how precisely the data pin it down).

## src/behavior_analysis/choice_logit/choice_regression.py::group_summary

Averages each weight across mice and tests whether the average differs from zero with a one-sample
t-test (mice as replicates). A significant positive weight on last value, for example, means mice
reliably prefer ports that paid more last time.

## src/behavior_analysis/task_statistics/cohort_comparison_stats.py

The numbers behind the figures that compare two cohorts (mean and standard error across mice). Kept
separate from the plotting so they can be tested.

## src/behavior_analysis/reversals/cumulative_reversal_removal.py

When you line up trials around a reversal and look further and further after it, sooner or later
the next reversal arrives and the later trials no longer reflect the first one. This file counts,
at each trial offset, how many reversal windows have already been cut short by the next reversal,
split by whether that next reversal was good, bad or unclassified. It tells you how far out the
averaged curves can be trusted.

## src/behavior_analysis/review/diagnostic_tables.py

Collects the diagnostic tables that analyses produce (poke summaries, port-finding checks, event
counts and so on) and stacks them across problems, one table per kind, with a column saying which
problem each row came from. These tables are for checking the data, not for results.

## src/behavior_analysis/review/diagnostic_tables.py::stack_diagnostics

Concatenates each kind of diagnostic table over the problems, so you get, for example, one poke
summary for the whole cohort instead of one per problem.

## src/behavior_analysis/data/drifting_task_rules.py

What the drifting task actually did, measured from the trial log rather than from the task code:
when block changes happened, which arm was set to zero (the one the mouse chose most, or least),
what the arm that had been at zero received, how often the payouts drifted between trials, and how
long blocks lasted. These numbers document the task for reading figures, and they set up the
simulated task so the model agents face the same rules as the mice.

## src/behavior_analysis/data/drifting_task_rules.py::block_changes

Finds every block change (a trial where a different arm pays zero than on the previous trial) and
records which arm was zeroed and what value the previously zero arm took on.

## src/behavior_analysis/data/drifting_task_rules.py::drift_counts

Counts how often a non-zero arm's payout changed between two consecutive trials within a block,
and how many chances it had to change. Their ratio is the drift rate. Session boundaries and gaps
are skipped, because an unknown number of drift steps may hide there.

## src/behavior_analysis/data/drifting_task_rules.py::complete_block_lengths

Lengths in trials of the blocks whose start and end were both seen (blocks cut by the start or end
of a problem are left out, since their true length is unknown).

## src/behavior_analysis/data/drifting_task_rules.py::block_change_summary

Per mouse, how often the task zeroed the most-chosen vs the least-chosen arm, and how often the old
zero arm inherited the zeroed arm's value. This checks that the task followed its intended rule.

## src/behavior_analysis/data/drifting_trial_table.py

Builds the one table that most drifting-task analyses start from: one row per trial, with the
mouse, problem, session, trial number, chosen port, the magnitude of every arm, the reward received
and the block. All the awkward details of the raw session data (several recordings per session,
session order, missing trials) are handled once, here.

## src/behavior_analysis/data/drifting_trial_table.py::drifting_trial_table

Converts the importer's session dictionaries into the trial table described above, in trial order
within each mouse's problem.

## src/behavior_analysis/data/exclude_mice.py

Lets you leave named mice out of the pooled drifting analyses (for example, a mouse with very few
trials). It removes them before pooling, so every analysis sees the same reduced set, and names the
extra output folder so the full and reduced results never mix.

## src/behavior_analysis/data/exclude_mice.py::exclude_mice_from_problems

Returns the problems with the named mice removed (and drops any problem left with no mouse). The
input is not changed.

## src/behavior_analysis/reversals/first_leave_fractions.py

Turns first-leave counts into fractions per mouse for plotting (see
`get_first_leave_after_good_reversals.py` for what "first leave" means).

## src/behavior_analysis/reversals/get_bad_reversal_info.py

Cuts out windows of trials around each bad reversal (a block change in the blocked task that
changes the payouts without changing which tower is best), for every mouse. It mirrors the
good-reversal version but keeps only reversals whose payouts just beforehand match a required
pattern, so the windows are comparable.

## src/behavior_analysis/reversals/get_bad_reversal_info.py::get_bad_reversal_info

For each mouse, finds the bad reversals, keeps those whose payouts before the reversal match the
required pattern, and returns the choices and rewards in a window before and after each one.

## src/behavior_analysis/data/get_believed_value.py

Estimates what the mouse "believes" each arm is worth, trial by trial, using the simplest rule: an
arm is worth what it paid the last time the mouse chose it. Arms the mouse has not tried yet in the
problem have no belief and are left out of the ranking. From these beliefs it ranks the arms (best,
second, third) and records the believed rank of the arm the mouse picked. Comparing believed ranks
with true ranks separates "the mouse is choosing badly" from "the mouse is choosing well given what
it has seen".

## src/behavior_analysis/data/get_believed_value.py::compute_believed_value

Walks through the trials in order and, before each trial, records every arm's believed magnitude
(its last payout), the believed best and second-best magnitudes, and the believed rank of the
chosen arm. Ties are recorded so later analyses can share credit between tied arms.

## src/behavior_analysis/reversals/get_choice_probs_around_good_reversals.py

The classic reversal plot: line up all good reversals at trial 0 and, at each trial before and
after, compute the fraction of times the mouse chose the old best tower, the new best tower and the
third tower. Averaging over reversals gives each mouse's curve; averaging mice gives the group
curve. A fast crossover after trial 0 means the mouse adapts quickly to the change.

## src/behavior_analysis/reversals/get_choice_probs_around_good_reversals.py::get_choice_probs_around_good_reversals

Computes those curves from the reversal windows: for each mouse, stacks one row per reversal of
which tower was chosen at each offset, averages within the mouse, then averages across mice (with
standard errors). Returns the trial offsets, the per-mouse curves and the group curve.

## src/behavior_analysis/reversals/get_choice_probs_around_good_reversals.py::apply_moving_average_to_choice_probs

Smooths the curves with a moving average, separately before and after the reversal so the sudden
change at trial 0 is not blurred across the boundary. Missing values are ignored rather than
treated as zero.

## src/behavior_analysis/reversals/get_choice_probs_around_good_reversals.py::remove_trials_after_bad_rev

If a bad reversal happens within a good reversal's window, the trials after it no longer belong to
the good reversal's conditions. This cuts each window short at the first such bad reversal.

## src/behavior_analysis/value/get_choice_probs_by_value.py

How does the choice depend on what the arms are worth? Grids P(choose best) and P(choose second) by
the value of the best arm (2–4) and the value of the second arm (1–3), using either true or believed
values. A mouse that is sensitive to value differences picks the best arm more often when the gap
between best and second is large. Counts are made per mouse and then averaged, so each mouse counts
equally.

## src/behavior_analysis/value/get_choice_probs_by_value.py::count_choice_probs

Counts, for each mouse and each (best value, second value) cell, how many trials fell in the cell
and how many times the mouse chose the best and the second arm. Beliefs reset at each problem
boundary.

## src/behavior_analysis/value/get_choice_probs_by_value.py::accumulate_choice_probs

Turns the counts into probabilities per mouse and averages them across mice, giving the heatmaps of
P(best) and P(second) by value.

## src/behavior_analysis/reversals/get_diagnostic_p_value.py

Simple paired statistics on rank choices across mice: is the best arm chosen more than the second,
and the second more than the third? "Paired" means each mouse is compared with itself, which removes
differences between mice.

## src/behavior_analysis/reversals/get_diagnostic_p_value.py::pvalue_paired_t_best_vs_second_vs_third

Averages each mouse's proportions of best, second and third choices over reversals, then runs
paired t-tests for each pair of ranks. Small p-values mean the mice reliably prefer one rank over
another.

## src/behavior_analysis/reversals/get_diagnostic_p_value.py::pvalue_paired_t_new_vs_third

After a good reversal, when the mouse first leaves the old best tower, does it go to the new best
more often than to the third? This paired t-test across mice answers that.

## src/behavior_analysis/reversals/get_first_leave_after_good_reversals.py

After a good reversal the old best tower stops being the best, and the mouse eventually leaves it.
Where does it go first: to the new best tower, or to the third one? Going to the new best more than
half the time suggests the mouse knows something about the other arms' values; a 50/50 split
suggests it leaves at random.

## src/behavior_analysis/reversals/get_first_leave_after_good_reversals.py::get_first_leave_after_good_reversals

For each reversal, finds the first trial after it on which the mouse chose a tower other than the
old best, and records whether that was the new best or the third. Returns counts per mouse.

## src/behavior_analysis/reversals/get_first_leave_after_good_reversals.py::average_first_leave_across_subjects

Turns each mouse's counts into fractions and averages them across mice with standard errors.

## src/behavior_analysis/reversals/get_good_reversal_info.py

The starting point for every reversal-aligned analysis: finds the good reversals in each mouse's
merged sessions and cuts out a window of trials before and after each one (choices and rewards per
tower and per rank). It also applies quality checks: reversals that do not show up in the payouts,
or where too many towers paid nothing beforehand, are dropped with a warning saying why.

## src/behavior_analysis/reversals/get_good_reversal_info.py::get_good_reversal_info

For every mouse, locates the good reversals (preferring the rig's own reversal counter, with
fallbacks), drops the ones that fail the quality checks, and returns the pre- and post-reversal
windows that the choice-probability, first-leave and rank analyses use.

## src/behavior_analysis/pokes/get_num_long_pokes.py

Counts "long pokes" per session: pokes held at a port other than the one chosen. A falling count
over sessions is a sign the mouse is learning where the offered ports are.

## src/behavior_analysis/rank_proportions/get_rank_counts_by_good_reversal.py

After each good reversal, how often did the mouse choose the best, second and third arm? Counts
choices by rank in each reversal's post window, either by true rank (what the arms actually paid)
or by believed rank (what the mouse last saw them pay).

## src/behavior_analysis/rank_proportions/get_rank_counts_by_good_reversal.py::get_rank_counts_by_good_reversal

Counts best, second and third choices by true rank in each reversal's post window, per mouse, and
converts them to proportions for the rank-proportion plots.

## src/behavior_analysis/rank_proportions/get_rank_counts_by_good_reversal.py::get_rank_counts_by_value

The same counts by true or believed rank. With believed ranks, an arm's rank is set before its new
payout is seen, arms not yet tried are left out, and tied arms share the credit equally.

## src/behavior_analysis/task_statistics/get_task_statistics.py

How long are blocks? Measures block lengths in trials from the positions of reversals (good, bad or
all), per mouse.

## src/behavior_analysis/task_statistics/get_task_statistics.py::get_block_lengths

Splits each mouse's trials at the reversal points and returns the length of each block, per mouse.

## src/behavior_analysis/reversals/get_total_reversals.py

Counts how many good, bad and total reversals each mouse experienced, and where they happened, using
the rig's counters (corrected for sessions that started and stopped).

## src/behavior_analysis/reversals/get_total_reversals.py::get_total_reversals

The total number of reversals for one mouse (good and bad separately when the rig logged them).

## src/behavior_analysis/reversals/get_total_reversals.py::get_all_reversal_indices

The trial numbers at which each mouse's good and bad reversals happened, across its merged sessions.

## src/behavior_analysis/task_statistics/get_transition_matrix.py

An empty placeholder for a future analysis of how mice move from one tower to the next.

## src/behavior_analysis/data/get_variables_across_sessions.py

Joins a mouse's sessions into one continuous record. The rig restarts its counters (reversals,
block number) every session, so simply gluing sessions together would make the counters jump back
to zero. This file offsets each session by where the previous one ended, so a block or reversal
count runs continuously across sessions.

## src/behavior_analysis/data/get_variables_across_sessions.py::get_vars_across_all_sessions

Merges every session of each mouse in date order into one record of per-trial values, with
reversal and block counters made continuous, and carries a reversal that happened at the very end of
a session into the next one.

## src/behavior_analysis/glm_hmm/glm_hmm.py

Are mice always using the same strategy, or do they switch between strategies? A GLM-HMM assumes
the mouse is, on each trial, in one of a few hidden "states", and in each state it chooses using its
own set of weights on the same evidence (value, win-stay/lose-shift signals, habit, port bias).
"Hidden" because we never see the state directly; we infer it from how well each state's weights
explain the choices around that trial. The mouse tends to stay in a state for a while (sticky
transitions), so states show up as stretches of trials with a consistent style of choosing. The
model is fitted with the EM algorithm, which alternates between guessing the state on each trial
and refitting each state's weights, from several starting points to avoid getting stuck. With one
state it is just an ordinary choice model.

## src/behavior_analysis/glm_hmm/glm_hmm.py::fit_restarts

Fits the GLM-HMM from several random starting points and keeps the best run, because EM can get
stuck in a poor solution depending on where it starts. To save time, every start runs briefly
first and only the most promising ones are run to the end.

## src/behavior_analysis/glm_hmm/glm_hmm.py::posteriors

For every trial, the probability that the mouse was in each state, given everything it did in the
session (before and after that trial). A trial with probability 0.9 for state 2 is confidently
assigned to state 2.

## src/behavior_analysis/glm_hmm/glm_hmm.py::dwell_times

How long, on average, the mouse stays in each state before switching, in trials. It comes straight
from the fitted chance of staying in the state from one trial to the next: if that chance is 0.98,
the mean stay is 1 / (1 − 0.98) = 50 trials.

## src/behavior_analysis/glm_hmm/glm_hmm.py::occupancy

The fraction of trials spent in each state (averaging the state probabilities over trials).

## src/behavior_analysis/glm_hmm/glm_hmm_features.py

Prepares one mouse's trials for the GLM-HMM: the evidence for each arm on each trial (from a fitted
kernel Q-learner, win-stay/lose-shift terms, habit and port bias), split into sessions (the state
process restarts at each session), plus the split into cross-validation folds. Whole sessions are
held out together, so a model is always tested on sessions it never saw.

## src/behavior_analysis/glm_hmm/glm_hmm_features.py::session_arrays

Builds the per-trial, per-arm evidence and the chosen arm for every session of one mouse, ready to
fit.

## src/behavior_analysis/glm_hmm/glm_hmm_features.py::assign_folds

Gives every session a fold number for cross-validation, dealing the sessions of each problem out in
turn so every fold has sessions from every problem.

## src/behavior_analysis/glm_hmm/glm_hmm_kernel.py

The choice-history kernel (how much choices 1 to 5 trials back push the mouse to repeat them) is
surprisingly flat in the mice. One explanation: the mouse switches between states, one that repeats
its last choice and one that doesn't, and pooling them looks flat. This file refits the choice
logit separately on the trials confidently assigned to each GLM-HMM state. If each state's kernel is
sharp, the mixture explains the flatness; if each state's kernel is still flat, it doesn't.

## src/behavior_analysis/glm_hmm/glm_hmm_kernel.py::kernel_by_state

Returns the choice-lag weights fitted to all trials and to the trials of each state, so their shapes
can be compared.

## src/behavior_analysis/glm_hmm/glm_hmm_recovery.py

Checks that the GLM-HMM can find states that really exist: fit it to simulated data where the true
states are known, then compare. Because fitted states come out in arbitrary order, each true state
is first matched to its closest fitted state, then the weights, the trial-by-trial state labels and
the dwell times are compared.

## src/behavior_analysis/glm_hmm/glm_hmm_recovery.py::match_states

Pairs each true state with the fitted state whose weights are most similar, using an optimal
one-to-one assignment.

## src/behavior_analysis/glm_hmm/glm_hmm_recovery.py::state_accuracy

The fraction of trials whose most likely fitted state is the right one (after matching). Near 1
means the model recovers when the simulated mouse switched.

## src/behavior_analysis/glm_hmm/glm_hmm_recovery.py::weight_recovery

Puts each true weight next to its recovered value for every matched state, for the recovery
scatter plots.

## src/behavior_analysis/glm_hmm/glm_hmm_recovery.py::weight_summary

How well the weights were recovered, in two numbers: the correlation between true and recovered
weights and their mean absolute difference.

## src/behavior_analysis/glm_hmm/glm_hmm_recovery.py::dwell_recovery

True and recovered average stay in each state, in trials.

## src/behavior_analysis/glm_hmm/glm_hmm_selection.py

How many states does a mouse need? More states always fit the training data better, so the honest
test is on held-out sessions. This file fits models with 1, 2, 3... states on some sessions, scores
them on the rest, and reports the *gain* of each model over the one-state model. To avoid choosing
extra states for tiny improvements, it picks the smallest number of states whose score is within one
standard error of the best.

## src/behavior_analysis/glm_hmm/glm_hmm_selection.py::cross_validate

Fits every number of states on all folds but one and scores it on the held-out fold, for every fold.

## src/behavior_analysis/glm_hmm/glm_hmm_selection.py::selection_table

Pools the held-out scores into nats per trial for each number of states, and the gain over one
state (positive means extra states help on unseen sessions).

## src/behavior_analysis/glm_hmm/glm_hmm_selection.py::select_n_states

Picks the number of states with the one-standard-error rule: the simplest model whose held-out score
is within one standard error of the best one.

## src/behavior_analysis/choice_logit/glm_utils.py

Helpers that build inputs for a multinomial logistic regression (a model predicting which of three
arms is chosen from choice and reward history), used by the older exploratory notebooks. Includes
mapping towers to arm labels and building the design matrices. The current notebooks use the choice
logit (`choice_logit.py`) instead.

## src/behavior_analysis/choice_logit/glm_utils.py::format_glm_input

Builds, for each mouse, the table of predictors (a constant, recent choices of each arm, recent
rewards, and their interactions) and the chosen arm on each trial, ready for the regression.

## src/behavior_analysis/pokes/long_pokes_by_session.py

Summarizes long pokes (pokes held at a port that was not the choice) session by session: the mean
and standard error across mice, plus each mouse's own series. Fewer long pokes over sessions suggest
the mouse is learning where to go.

## src/behavior_analysis/pokes/long_pokes_by_session.py::summarize_pokes_by_session

Per session, the mean and standard error of long pokes across mice, with each mouse's series.

## src/behavior_analysis/pokes/long_pokes_by_session.py::to_relative_sessions

Renumbers each mouse's sessions 1, 2, 3... within the problem, so mice that started on different
dates can be compared session by session.

## src/behavior_analysis/data/matched_trial_budget.py

Mice do different numbers of trials on a problem, which can bias comparisons (a mouse with more
trials has had more chance to learn). This file keeps only each mouse's first N trials, so every
mouse is compared on the same amount of experience.

## src/behavior_analysis/data/matched_trial_budget.py::match_trial_budget

Trims each mouse to its first N trials, where N is either given or set from a reference mouse, and
reports which mice had fewer trials than N (they keep everything they have).

## src/behavior_analysis/data/merge_across_problems.py

Helpers for pooling per-problem results (reversal windows, first-leave counts) across problems for
the same mouse, so analyses can be averaged across a mouse's whole training.

## src/behavior_analysis/value/model_value_psychometrics.py

Psychometric curves ask: as the value of an option goes up, does the mouse choose it more? The
simple version uses whole-pulse values (best 2–4, second 1–3). This file does the same with
continuous values, such as the choice logit's model values or a Q-learner's learned values. Those
values have no fixed scale and differ between mice, so each mouse's trials are split into bins
holding equal numbers of trials (quantile bins) before averaging.

It also draws a **baseline**: the curve predicted by everything except value (habit, tower bias).
Mice prefer some towers and repeat choices, and that alone can make P(best) rise with value. Only
the part of the curve above the baseline is evidence that value guides choice.

## src/behavior_analysis/value/model_value_psychometrics.py::rank_table

For each trial, sorts the three arms' values from highest to lowest and records how much of the
mouse's choice went to the top two. Used to place trials on the psychometric axes.

## src/behavior_analysis/value/model_value_psychometrics.py::psychometric_curves_from_table

Computes P(choose best) and P(choose second) against the best value, the second value and their
difference, binned per mouse and then averaged, together with the baseline curves.

## src/behavior_analysis/value/model_value_psychometrics.py::value_effect

One number for "does value matter beyond the baseline": how much more P(best) rises from the lowest
to the highest value bin than the baseline predicts, averaged over mice. About 0 means value adds
nothing; positive means value guides choice.

## src/behavior_analysis/value/model_value_psychometrics.py::stay_switch_pairs_from_values

Builds pairs of consecutive trials and records, for the second trial, the value of the arm the mouse
was on, the value of the best alternative, and whether it stayed or switched. These pairs feed the
stay/switch curves.

## src/behavior_analysis/value/model_value_psychometrics.py::stay_switch_curves_by_reward

P(stay) and P(switch) against value, drawn separately depending on what the previous trial paid
(nothing vs something), with the baseline. Splitting by the last payout matters because mice treat
"got nothing" very differently from "got a little".

## src/behavior_analysis/value/model_value_psychometrics.py::switch_target_curves_by_reward

When the mouse does switch, does it go to the better of the two other arms? Plots P(choose the
better alternative) against how much better it is, per previous-payout condition. Switches between
two equally valued alternatives are left out, since neither is "better".

## src/behavior_analysis/switching/no_second_best.py

Lists the blocks in which two or more arms paid nothing at the same time. In those blocks there is
no second-best arm, so analyses that compare the best with the second best have nothing to compare.
This is a table for checking the data by hand.

## src/behavior_analysis/review/no_trial_sessions.py

A table of the files and sessions the importer skipped because they had no usable trials, with the
reason, so you can check nothing important was lost.

## src/behavior_analysis/data/outcome_events.py

Reads the exact time each trial's outcome reached the mouse from the raw pyControl log: the start of
the first reward pulse on a rewarded trial, or the start of the no-reward cue otherwise. These times
are what imaging and photometry responses are aligned to, so they need to be precise.

## src/behavior_analysis/data/outcome_events.py::load_outcome_events

Reads one pyControl `.tsv` file and returns a table with one row per trial: whether it was
rewarded, how many pulses, and the outcome time.

## src/behavior_analysis/pokes/poke_counts.py

Counts pokes at each of the nine ports, per session, mouse and problem: pokes between starting a
trial and choosing, and pokes after the choice. Pokes at ports that are not offered show where the
mouse thinks the reward might be, for example ports from its previous problem.

## src/behavior_analysis/pokes/poke_counts.py::mouse_poke_counts

Poke counts for every session of one mouse. A session whose raw log cannot be parsed is skipped with
a warning rather than stopping the whole run.

## src/behavior_analysis/pokes/poke_counts.py::previous_problem_choice_towers

For each mouse and problem, the towers offered in the problem before. Used to check whether mice
keep visiting their old towers after a problem change.

## src/behavior_analysis/pokes/poke_events.py

Reads pokes from the raw pyControl event log, trial by trial: the long pokes (held long enough to
count) between initiating a trial and choosing, and every poke after the choice until the next
trial starts. Other files count and plot these.

## src/behavior_analysis/pokes/poke_tables.py

Tables summarizing the poke analysis per session (trials counted, dropped, port-finding trials,
and whether raw and logged long-poke counts agree), for checking the data.

## src/behavior_analysis/pokes/port_finding_check.py

At the start of a problem, the mouse goes through "port finding": it must find each offered port a
set number of times. So every mouse should have at least (number of ports × that threshold)
port-finding trials. This check flags mice with fewer, which usually means a session is missing
from the data or the counts were set by hand.

## src/behavior_analysis/pokes/port_finding_sessions.py

A mouse that does not finish port finding in one session continues it in the next. A session with
only port finding has no bandit trials, so on its own it belongs to no problem. This file assigns
such sessions to the problem that follows them, so their pokes count toward that problem.

## src/behavior_analysis/pokes/port_finding_sessions.py::assign_port_finding_sessions

Assigns each port-finding-only session to the next session with bandit trials, if that session has
the same towers, and warns about any session that cannot be placed.

## src/behavior_analysis/pokes/port_finding_summary.py

Where do mice poke during port finding and just after it? Counts pokes per port in three windows:
during port finding, in the first trials of the problem, and in the first blocks. The results are
grouped by maze layout (which towers are offered now and in the previous problem), to see whether
mice search near their old towers.

## src/behavior_analysis/pokes/port_finding_summary.py::mean_percent_map

For one window, each port's share of the pokes, averaged across mice, which is what the maze
heatmaps show.

## src/behavior_analysis/pokes/port_finding_summary.py::category_fraction_table

The share of pokes going to each category of port (current towers, previous-problem towers, others)
per mouse, problem and window.

## src/behavior_analysis/pokes/port_preference.py

Is each port chosen as often as it deserves? The preference of a port is P(chosen) / P(best): how
often it was chosen divided by how often it was actually the best. A value of 1 means it is chosen
exactly as often as it is best; above 1, the mouse over-chooses it (for example, a port close to the
start); below 1, it under-chooses it. Ports are also grouped into categories by their position
relative to the start port, to look for spatial biases.

## src/behavior_analysis/pokes/port_preference.py::count_port_preference

For each mouse and problem, counts how often each port (and category) was offered, chosen and best.

## src/behavior_analysis/pokes/port_preference.py::category_summary

Mean and standard error across mice of each category's preference ratio.

## src/behavior_analysis/data/problem_selection.py

Decides which drifting problems go into an analysis. Cohort 1's behavior degrades after drifting
problem 7 (two mice stop there, and the three that continue come back from surgery), so its problems
are cut off at 7; cohort 3 keeps all of them. The cutoff is a single setting.

## src/behavior_analysis/data/problem_selection.py::select_problems

Keeps the trials of the problems up to each cohort's cutoff.

## src/behavior_analysis/task_statistics/problem_summary.py

Summaries across problems for each mouse: how many sessions it needed per problem, the average
block length, and the average number of good reversals per session. Falling sessions-per-problem
across training means the mice get faster at learning new problems.

## src/behavior_analysis/data/problem_utils.py

General helpers for working with per-problem data: turning problem labels into numbers, joining a
mouse's sessions, counting trials and summarizing values across mice.

## src/behavior_analysis/rank_tests/rank_calibration.py

A number from one mouse (a weight, a slope) is hard to judge alone, because the task's schedule and
the analysis itself can produce non-zero values even from a mouse that does nothing special. So
each mouse is compared with a **null**: a model agent fitted to that mouse and replayed many times
on that mouse's own schedule, analyzed the same way. The z-score says how many standard deviations
the mouse lies from its agent's replays, and the empirical p is the fraction of replays as extreme as
the mouse. Across mice, the z-scores are averaged and tested.

## src/behavior_analysis/rank_tests/rank_calibration.py::calibrate

For every mouse and readout, compares the observed value with its replays: replay mean and spread,
z-score and empirical p.

## src/behavior_analysis/rank_tests/rank_calibration.py::summarize

Across mice: mean and standard error of the z-scores, a Wilcoxon test of whether they are above
zero, and which mice individually pass.

## src/behavior_analysis/rank_tests/rank_crossings.py

Test 2 of the ranked value algorithm: does the mouse switch abruptly when it comes to believe that
another arm has become better than the one it is on? A ranked algorithm leaves as soon as the ranks
cross (a step), while a standard value learner leaves gradually, more often the bigger the
difference. The file measures P(leave) against the believed difference between the current arm and
the best alternative, and compares a smooth curve with a step at zero. It also lines up trials
around the moments the ranks cross, to see how fast the mouse adopts the new best.

## src/behavior_analysis/rank_tests/rank_decisions.py

The building block of the rank tests: a table of stay-or-leave decisions. Each row is a pair of
consecutive trials, with the believed values of all arms at the first one and whether the mouse left
on the second. Only decisions where all three arms have been tried are included, so no rank rests on
a guess.

## src/behavior_analysis/rank_tests/rank_logit.py

A small, fast logistic regression used by the rank tests. Those tests are run on every mouse and on
hundreds of simulated replays of each, so they need a solver that is quick for small problems. A
mild penalty keeps weights finite even when a feature predicts the outcome perfectly.

## src/behavior_analysis/rank_tests/rank_logit.py::fit_logit

Fits P(outcome) = sigmoid(features × weights) and returns the weights with their uncertainty.

## src/behavior_analysis/rank_tests/rank_logit.py::cross_validate

Fits on all problems but one and scores on the one left out, for each problem, summing the held-out
log likelihood. It stops with an error, rather than skipping silently, if a training set has only one
outcome.

## src/behavior_analysis/rank_proportions/rank_proportions.py

Summaries of how often mice choose the best, second and third arm: means and standard errors per
mouse or per block. Each mean is paired with a standard error over the same units it averages.

## src/behavior_analysis/rank_proportions/rank_proportions_summary.py

The across-mouse summary of best, second and third choice proportions around good reversals.

## src/behavior_analysis/rank_tests/rank_result.py

The common shape of a rank test's result: single numbers (readouts) that get compared with replays
of fitted agents, and curves that get drawn. Keeping one shape lets every test be calibrated and
plotted the same way.

## src/behavior_analysis/rank_tests/rank_signatures.py

Meg's thesis tests a **ranked value algorithm**: the mouse ranks the options (best, second, third)
and its choices depend on comparisons between neighboring ranks. A standard reinforcement learner
instead holds a value for each arm and chooses with probabilities that rise smoothly with value (a
softmax). Both prefer better arms, so the tests look for behavior only the ranked algorithm
predicts:

1. Staying depends on the second-best option, not the third.
2. Switching is abrupt when ranks cross.
3. The worst option is treated as a category, chosen at a floor rate whatever its value.
4. Ranks predict choice better than values.
5. Mice transfer knowledge across problems and infer the new best at block changes.

This file runs all five on one mouse (or one simulated replay).

## src/behavior_analysis/rank_tests/rank_signatures.py::analyze_mouse

Runs every rank test on one mouse's trials and returns all their readouts and curves, ready to be
calibrated against replays of fitted agents.

## src/behavior_analysis/rank_tests/rank_state.py

What the mouse believes on each trial, as used by all rank tests: each arm's believed value (its last
payout in the current problem, carried across sessions) and the arms' ranks. A trial's state only
counts once all three arms have been tried, so ranks never rest on a guess.

## src/behavior_analysis/rank_tests/rank_state.py::mouse_state

Builds that per-trial state (believed values, ranks, session and problem boundaries) for one mouse.

## src/behavior_analysis/rank_tests/rank_stay.py

Test 1 and part of test 4, using models of P(stay). The key comparison: a ranked algorithm decides
whether to stay by comparing the current arm with the *best* alternative only, so the *worst*
alternative's value should not matter. A softmax learner is pulled by both alternatives, so both
values should lower staying. The models also include habit terms (how long the mouse has been on the
arm, its recent choices, port bias), so the value weights are not just habit in disguise. The file
also asks whether ranks or values describe staying better, and whether staying depends on the
absolute size of a payout at a fixed rank.

## src/behavior_analysis/rank_tests/rank_switch.py

Part of test 4: when the mouse leaves an arm, which of the two alternatives does it go to? The
switch regression is fitted with the alternatives' believed values, their ranks, or both, and the
versions are compared on held-out problems. A value learner should be described better by values,
a rank learner by ranks.

## src/behavior_analysis/rank_tests/rank_third.py

Test 3: is the arm believed worst treated as a category? A ranked algorithm chooses the worst arm at
a low, constant rate, whatever its value (it only moves up when the arm above it falls below it). A
softmax learner chooses it less the further its value lies below the second best. The readout is
the slope of P(choose the worst arm) against that gap: flat for ranks, negative for a softmax.

## src/behavior_analysis/rank_tests/rank_transfer.py

Test 5. **Transfer**: a mouse that has learned the task's structure should get faster at each new
problem; one that starts each problem from scratch should not. Measured as the slope of early
performance against problem number, compared with replays of a fitted agent, which cannot transfer
and so show the schedule's own trend. **Inference**: at a block change the task zeroes the arm the
mouse chose most and gives its value to the arm that was at zero. A mouse that knows this rule can
work out the new best without sampling it: when its favorite arm suddenly pays nothing, the arm it
had been avoiding now holds the favorite's old value. A mouse going only by last payouts still
thinks that arm is worth 0 and goes to the third arm instead. The readout is the share of first
leaves after such a change that go to the old zero arm.

## src/behavior_analysis/rank_tests/rank_worst_split.py

A follow-up to test 1. Mice stay *more* when the worst alternative is believed to be better, which
neither a ranked algorithm nor a softmax predicts. Because one arm always pays zero, some beliefs
must be out of date (an arm believed to pay may have turned to zero since the mouse last visited
it). This file splits the decisions by whether the beliefs are consistent with exactly one zero arm,
to see how much of the odd weight comes from those stale beliefs.

## src/behavior_analysis/reversals/reversal_aligned_arms.py

In the drifting task: around each block change, how often does the mouse choose each arm, and what
does each arm pay? The arms are labeled by their rank just before the change (or just after), so
you can follow, say, the arm that was best before the change. Curves are averaged over block
changes within a mouse and then across mice (each mouse counts once), or over all block changes
together (each event counts once).

## src/behavior_analysis/reversals/reversal_aligned_arms.py::build_windows_with_magnitudes

Cuts a window of trials around each block change, staying within the blocks on either side, and
adds every arm's payout over the same trials.

## src/behavior_analysis/reversals/reversal_aligned_arms.py::compute_reversal_aligned_arms

Labels the arms by rank, keeps one type of block change, and averages choices and payouts per arm
label into curves, either per mouse then across mice, or across all events.

## src/behavior_analysis/reversals/reversal_alignment.py

Puts windows of different lengths on a shared trial axis so they can be averaged: trials before the
event end at −1, the event trial is 0, and missing positions are left empty rather than filled with
zeros. Also handles the smoothing.

## src/behavior_analysis/reversals/reversal_arm_ranks.py

Ranks the three arms by payout just before and just after a block change, labels them, and sorts
block changes into types (for example "the old best became the worst"). When arms are tied, the
order is genuinely unknown, so a block change whose type depends on how a tie is broken is marked
ambiguous instead of being guessed.

## src/behavior_analysis/reversals/reversal_arm_ranks.py::classify_reversal_type

The type of one block change, such as "the best arm became third" or "the second arm became third",
or "ambiguous" if ties allow more than one answer.

## src/behavior_analysis/reversals/reversal_format.py

Older sessions record only block changes; newer blocked-task sessions also record whether each was a
good or a bad reversal. This file works out which format each session uses and checks that analyses
only get the format they expect.

## src/behavior_analysis/reversals/reversals_by_session.py

Reversal counts session by session, with cumulative and smoothed curves. More reversals per session
means the mouse is completing blocks faster.

## src/behavior_analysis/reversals/reversals_by_trial.py

Reversal counts against the number of completed trials rather than sessions, so mice that do
different numbers of trials per session can be compared.

## src/behavior_analysis/review/review_context.py

Labels each session in the review tables with its task type and problem, including sessions the
importer left out (they get the problem the mouse was on before them).

## src/behavior_analysis/switching/second_best.py

When a mouse leaves an arm, does it go to the better of the two alternatives? Three versions of the
question:

- after leaving the best arm: does it go to the second best rather than the worst?
- after leaving any arm: does it go to the better alternative?
- a graded version: is it more likely to pick the better alternative the bigger the difference?

Each is split by whether the mouse just got nothing or something. Because chance is not exactly 50%
here (the schedule and habits shape it), each mouse is compared with replays of fitted agents on its
own schedule.

## src/behavior_analysis/switching/second_best.py::analyze_mouse

Computes all three measures and the graded curves for one mouse, after a zero payout, after a
non-zero payout and pooled.

## src/behavior_analysis/switching/second_best.py::calibrate

Compares each mouse's proportions with the same proportions in replays of an agent fitted to that
mouse: z-score and empirical p.

## src/behavior_analysis/switching/second_best.py::calibrated_summary

Across mice: mean and standard error of the calibrated z-scores and Wilcoxon and sign tests of
whether they are above zero.

## src/behavior_analysis/task_statistics/sessions_per_problem.py

How many sessions each mouse needed to complete each drifting problem, counting split sessions once
and port-finding-only sessions separately. A mouse's latest problem may still be running, so it is
marked incomplete and left out of the averages.

## src/behavior_analysis/task_statistics/sessions_per_problem.py::sessions_per_problem_table

One row per mouse and drifting problem with the number of sessions it took.

## src/behavior_analysis/task_statistics/single_session.py

The numbers behind the single-session summary figure: what was chosen on each trial, what each arm
paid, and where block changes happened.

## src/behavior_analysis/reversals/split_early_late_good_reversals.py

Splits each mouse's good reversals into early ones (the first N) and late ones (the rest), so you
can compare how mice handled reversals while still learning the task with how they handled them
once experienced.

## src/behavior_analysis/reversals/split_early_late_good_reversals.py::split_good_reversals_early_late

Sorts each mouse's valid good reversals in order and returns two sets: the first N, and the rest.

## src/behavior_analysis/reversals/split_early_late_sessions.py

An empty placeholder for splitting sessions into early and late.

## src/behavior_analysis/reversals/split_good_reversals_by_best_change.py

After a good reversal, the old best arm either drops to second place or all the way to third. These
two cases may be handled differently by the mouse (leaving a merely worse arm vs a now-worst one),
so this file splits the reversals into the two groups.

## src/behavior_analysis/task_statistics/starting_magnitudes.py

Payouts carry over from one session of a problem to the next. A session that starts with different
payouts from those the previous session ended on means something changed in between (for example,
settings were edited). This lists such sessions for review.

## src/behavior_analysis/data/stats.py

Shared summary statistics. The key rule: a standard error needs at least two values, so with fewer
it is reported as missing (NaN), never as zero. A zero error bar would wrongly suggest certainty.

## src/behavior_analysis/data/stats.py::mean_se_n

The mean, standard error and number of values, ignoring missing values. The standard error is the
standard deviation divided by the square root of n: how precisely the mean is known.

## src/behavior_analysis/value/stay_by_run_length.py

Does staying get more likely the longer the mouse has been on one port? A *run* is a stretch of
consecutive choices of the same port. If P(stay) rises with run length, the mouse builds momentum
(a habit) the longer it stays, beyond what the last reward says. Shown separately after a zero and
after a non-zero payout.

## src/behavior_analysis/value/stay_by_run_length.py::run_lengths

For every trial, how many trials in a row the mouse had chosen the same port just before it.

## src/behavior_analysis/value/stay_by_run_length.py::stay_by_run_length

P(stay) in bins of run length (1, 2, 3, 4–5, 6–9, 10+), after a zero and after a non-zero payout.

## src/behavior_analysis/value/stay_by_run_length.py::run_length_slopes

Per mouse, how steeply P(stay) rises across the run-length bins: one number for "how much does
momentum build".

## src/behavior_analysis/value/stay_switch_psychometrics.py

The basic stay-or-switch question: given what the current arm is worth and what the best
alternative is worth, how likely is the mouse to stay? Values can be the true payouts or the
believed ones (what each arm paid last time). The file also asks, when the mouse switches, whether
it picks the better alternative, and makes a heatmap of P(stay) by current value and alternative
value.

## src/behavior_analysis/value/stay_switch_psychometrics.py::stay_switch_pairs

Every pair of consecutive trials, with the values of the current arm and the alternatives and
whether the mouse stayed. Trials with ties are counted so they can be reported.

## src/behavior_analysis/value/stay_switch_psychometrics.py::stay_switch_curves_by_row

P(stay) curves drawn separately for each value of the current arm.

## src/behavior_analysis/value/stay_switch_psychometrics.py::switch_target_curves_by_row

When the mouse switches, P(choose the better alternative), per value of the current arm.

## src/behavior_analysis/value/stay_switch_psychometrics.py::stay_heatmap

A grid of P(stay) over (current value, best alternative value), per mouse and averaged across mice.

## src/behavior_analysis/strategy_models/strategy_models.py

Which kind of strategy best describes each mouse? Several candidate models are fitted and scored on
problems they were not fitted to:

- **WSLS (Magnitude)**: win-stay/lose-shift, where the chance of staying grows with the size of the
  last payout, and switches go to recently chosen arms.
- **Heuristic plus**: the same, with switches that also consider what the alternatives last paid.
- **Q-learner**: a value learner that keeps a running estimate of each arm's worth.
- **Kernel + forgetting Q-learner**: adds habit (a choice kernel) and forgetting of unchosen arms.
- **WSLS (Habit)**: WSLS (Magnitude) plus a habit term.

The model with the lowest held-out nats per choice describes the mouse best. The file also checks
that the comparison is fair by simulating each model and seeing whether the fitting recovers the
right model (model recovery) and its parameters (parameter recovery).

## src/behavior_analysis/strategy_models/strategy_models.py::cross_validate

For one mouse, fits each model on all problems but one, scores it on the one left out, for every
problem; also returns the parameters fitted to all trials.

## src/behavior_analysis/strategy_models/strategy_models.py::nats_table

Held-out nats per choice of each model and mouse (lower is better).

## src/behavior_analysis/strategy_models/strategy_models.py::heldout_nats_per_trial

The same scores, adjusted so they can be compared with models scored on every trial (the first trial
of a session gets a 1-in-3 guess).

## src/behavior_analysis/strategy_models/strategy_models.py::difference_table

Per mouse, the difference in held-out score between pairs of models. Negative means the first model
predicts better.

## src/behavior_analysis/strategy_models/strategy_models.py::difference_summary

Across mice: the mean difference with its standard error and paired tests (Wilcoxon and sign) of
whether one model reliably beats the other.

## src/behavior_analysis/strategy_models/strategy_models.py::winners

The best model for each mouse.

## src/behavior_analysis/strategy_models/strategy_models.py::recovery_matrix

Model recovery as a table: for mice simulated by each model (rows), how often each model (columns)
wins the comparison. A good comparison has most counts on the diagonal: data from model X are best
described by model X.

## src/behavior_analysis/strategy_models/strategy_models.py::parameter_recovery_summary

How well fitting recovers known parameters from simulated mice: the average bias, the typical error
(root-mean-square) and the correlation between true and estimated values.

## src/behavior_analysis/strategy_models/strategy_models.py::decay_profile

How the held-out score of WSLS (Magnitude) changes as its memory of recent choices (the trace decay)
is varied. The best decay says over how many trials recent choices still steer switches.

## src/behavior_analysis/review/suspect_reversal_table.py

Lists, for review by hand, the reversals and block changes that the analyses dropped because they
did not show up in the payouts, with what is needed to find them in the raw files.

## src/behavior_analysis/review/suspect_reversals.py

Counts, beside the rig's own totals, how many logged good reversals in the blocked task did not
actually change which tower was best, using the same rule as the analyses.

## src/behavior_analysis/review/suspect_sessions.py

Lists sessions where a mouse may have been run on the wrong settings: a single odd session in the
middle of a problem, settings changed within a session, and similar flags, for review.

## src/behavior_analysis/switching/switch_regression.py

Where does a mouse go when it leaves an arm? On a switch trial, the mouse has two arms to choose
from (the two it was not on). This model predicts which of the two it picks from:

- **choice history**: how recently and how often it chose each of them (habit);
- **value**: what each is worth, as the true payouts, the believed values (last payouts) or the
  choice logit's model values.

The question is whether value adds anything beyond habit. The value weight says how strongly the
mouse goes to the better alternative; the held-out gain says how much value improves prediction on
unseen problems. Results are split by whether the mouse just got nothing (a "loss") or something.
Because habit and the schedule alone can produce a value weight, each mouse is also compared with
replays of agents that never look at value (the null calibration).

## src/behavior_analysis/switching/switch_regression.py::switch_data

Picks out one mouse's switch trials, with the choice-history features and the values of both
alternatives.

## src/behavior_analysis/switching/switch_regression.py::analyze_mouse

For one mouse: chooses how fast the choice-history memory fades (the trace decay), then fits and
cross-validates every model, with and without value, after a zero and after a non-zero payout.

## src/behavior_analysis/switching/switch_regression.py::value_table

Per mouse and condition, the value weight (log-odds per pulse of choosing the better alternative)
and how much value improves held-out prediction.

## src/behavior_analysis/switching/switch_regression.py::group_summary

Across mice: mean and standard error of the value weight and the gain, with Wilcoxon and sign tests.

## src/behavior_analysis/switching/switch_regression.py::null_calibration

Compares each mouse's value weight and gain with the same numbers from agents fitted to that mouse
that ignore value, replayed on its schedule. A z-score well above zero means the mouse uses value
more than habit and schedule alone can explain.

## src/behavior_analysis/switching/switch_regression.py::calibrated_summary

Across mice: mean and standard error of the calibrated z-scores with Wilcoxon and sign tests.

## src/behavior_analysis/task_statistics/trial_block_summary.py

Tables of trials per session and per block for the cohort summaries, with blocks continuing across
session boundaries and blocks cut by the start or end of the data flagged.

## src/behavior_analysis/task_statistics/trial_block_summary.py::build_trial_block_tables

Builds the per-session and per-block trial tables.

## src/behavior_analysis/task_statistics/trial_count_summaries.py

Summaries of trials per session or block for the cohort figures (per mouse, per problem, over time).

## src/behavior_analysis/value/value_change_complete_window.py

The value-change curves normally keep every event and cut its window short when something else
changes, so far from the event fewer events contribute. This version keeps only events whose whole
window (for example 2 trials before to 4 after) is clean, so every point of the curve averages the
same events. It trades the number of events for a curve that is easier to read.

## src/behavior_analysis/value/value_change_complete_window.py::select_complete_windows

Keeps the events with a complete, clean window and counts why the others were excluded.

## src/behavior_analysis/value/value_change_diagnostic.py

For checking one plot by eye: for each best-value decrease in a session, which trials the analysis
counted and which arm it labeled as best, second and third.

## src/behavior_analysis/value/value_change_mouse_curves.py

A table of each mouse's curve behind the mouse-weighted value-change plot, with how many events
contribute at each trial, so the plotted mean can be checked.

## src/behavior_analysis/value/value_change_windows.py

In the drifting task, payouts change gradually. Does the mouse notice? This file finds the moments
when the true payout of the best (or second-best) arm went up or down, and lines up the mouse's
choices around them. If the mouse tracks values, it should move away from the best arm soon after
its payout drops, especially when the drop makes it fall below the second (a *rank swap*). Windows
stop at block changes and session edges, and can stop at the next rank swap, so each window reflects
one change.

## src/behavior_analysis/value/value_change_windows.py::get_value_change_windows

Finds every change of the chosen rank's true payout in the chosen direction and returns the choices
in a window around each one.

## src/behavior_analysis/value/value_change_windows.py::truncate_windows_at_rank_swap

Ends each window at the next time the best and second arms swap places, so later trials that belong
to a different ranking are not counted.

## src/behavior_analysis/value/value_change_windows.py::summarize_windows

Averages the windows into curves (P(choose each arm) at each trial from the event), per mouse and
across mice.

## src/behavior_analysis/value/value_change_windows.py::contamination_summary

How many windows contain a further payout change or a rank swap, which says how clean the windows
are.

## src/behavior_analysis/value/value_psychometrics.py

P(choose best) and P(choose second) against the arms' values in whole pulses: the best arm's value,
the second's, and their difference. A mouse sensitive to value chooses the best more often when it
is worth more and when the gap to the second is bigger. Mice are the replicates: every mean and
standard error is across mice.

## src/behavior_analysis/value/value_psychometrics.py::psychometric_curves

Builds the curves of P(best) and P(second) against the best value, second value and their
difference, from per-mouse counts.

## src/behavior_analysis/value/value_psychometrics.py::heatmap_tables

The probabilities and trial counts behind the value heatmaps, overall and per mouse.

## src/behavior_analysis/data/warnings_log.py

Keeps one row per distinct warning raised during a cohort run (which otherwise scroll past in the
console), with how often it happened and which mouse or session it names.

## src/agents/best_rl.py

"Best RL" is a published reinforcement-learning model (from the Eckstein et al. work this repo
compares against), reimplemented as a simulated mouse. Each trial it picks an arm with a softmax over
its values plus a bonus for repeating the previous arm; after the outcome it moves the chosen arm's
value toward the reward and lets every value fade back toward a starting level (forgetting). It is
fitted to the whole data set by maximum likelihood rather than by the paper's training procedure.

## src/agents/best_rl.py::BestRLMouse

The Best RL model as a simulated mouse: it can choose an arm and update its values, so it can play
the simulated task.

## src/agents/choice_logit_truth.py

A check on the choice logit. When a simulated value learner (whose rules we know exactly) is
analyzed with the choice logit, what weights *should* come out? This file works out the answer: it
replays the simulated animal to recover its true choice scores, then finds the logit weights closest
to them. Comparing those with the weights the logit actually fits shows whether the logit recovers a
known learner correctly. It also gives the range of reward weights a value learner can produce at
each lag, since a reward fades with later visits to the same arm, not with trials.

## src/agents/choice_logit_truth.py::true_utilities

Replays one simulated value learner's trials and returns, on every trial, the true choice score of
each arm and its learned value.

## src/agents/choice_logit_truth.py::generating_weights

The choice-logit weights that best match the simulated learner's true choice probabilities: the
answer the logit should recover.

## src/agents/choice_logit_truth.py::q_kernel_bounds

For each lag, the smallest and largest reward weight a value learner can imply, depending on how
often the arm was chosen since the reward.

## src/agents/cohort_fitting.py

Fits a Q-learning model to each mouse and tests it on whole problems it was not fitted to. Each fold
is saved, so a long run can be stopped and resumed. Sessions with missing or invalid trials are
rejected, so a mouse's learning history is never joined across a gap.

## src/agents/cohort_fitting.py::prepare_cohort_sequences

Converts the sessions into the arrays the model needs and returns an audit table of every session,
saying which were kept and why others were rejected.

## src/agents/cohort_fitting.py::run_problem_cross_validation

For each eligible mouse, fits the model on all problems but one and scores it on the one left out,
saving each fit so reruns pick up where they left off.

## src/agents/drifting_simulation.py

Lets fitted model agents play the drifting task the same way the mice did: on each mouse's own
problems, with the same towers and trial counts, and with the payouts following the task's rules
(drift rate and block changes measured from the data). Running a fitted agent through the task
shows what behavior that agent produces, so the mice's behavior can be compared with it. It also
compares two versions of the basic value learner: one that learns from the number of pulses and one
that only learns whether it got any reward.

## src/agents/drifting_simulation.py::fit_agent

Fits the four-parameter value learner to one mouse's choices, learning either from pulse counts or
from rewarded-or-not.

## src/agents/drifting_simulation.py::fits_table

The fitted agents as a table, with how well each explains the mouse's choices and the difference
between the pulse-learning and binary-learning versions.

## src/agents/drifting_simulation.py::calibrate_drift_rate

Finds the drift setting for the simulated task that reproduces how often payouts change in the real
data.

## src/agents/drifting_simulation.py::mouse_layouts

What a simulation copies from each of a mouse's real problems: towers, sessions and trial counts.

## src/agents/drifting_simulation.py::simulate_policy_mouse

Plays any agent (a "policy") through all of one mouse's problems, starting fresh at each problem.

## src/agents/drifting_simulation.py::simulate_mouse

Plays one fitted value agent through all of one mouse's problems.

## src/agents/glm_hmm_agent.py

The evidence the GLM-HMM uses, and a simulated mouse that switches between strategies. On each trial
every arm gets a set of signals computed from history alone: its value in a kernel Q-learner, its
habit trace, whether it is the arm just chosen and paid (with how much), whether it just paid
nothing, how recently it was chosen, and whether it is a "plus" port near the start. A strategy is
then just a set of weights on those signals: a value strategy weights value, a win-stay/lose-shift
strategy weights the stay signals, a habit strategy weights habit and port bias. The switching agent
holds a hidden state that changes now and then and uses that state's weights, which is how the
simulated data for testing the GLM-HMM are made.

## src/agents/glm_hmm_agent.py::value_weights

The weights that make a strategy behave like the fitted kernel Q-learner.

## src/agents/glm_hmm_agent.py::wsls_weights

The weights that make a strategy behave like win-stay/lose-shift with magnitude.

## src/agents/glm_hmm_agent.py::habit_weights

The weights of a habit or biased strategy: repeat recent choices and prefer the plus ports.

## src/agents/glm_hmm_agent.py::blend_weights

A fixed mixture of the strategies' weights used on every trial. It is a no-switching control: a
mouse that blends strategies without switching between them.

## src/agents/glm_hmm_agent.py::FeaturePolicy

A simulated mouse with hidden states: at each session start it draws a state, after every trial it
may switch state, and it chooses with that state's weights. With one state it never switches.

## src/agents/heuristic_agent.py

A step up from win-stay/lose-shift. The agent first decides whether to stay, exactly like WSLS
(Magnitude): bigger last payouts make staying more likely. If it switches, it chooses between the two
other arms using how often it chose each recently, what each paid last time, and whether it has ever
tried it. It uses some value information, but in a simple rule-based way rather than by learning
values.

## src/agents/heuristic_agent.py::fit

Fits the heuristic agent to one mouse's choices: the stay rule and the switch rule are fitted
separately.

## src/agents/heuristic_agent.py::log_likelihood

How probable the mouse's actual choices are under the heuristic agent, in log units (higher is
better).

## src/agents/kernel_q_agent.py

The main value-learning model (after Aloor et al.). It keeps a learned value Q for each arm and a
habit trace K for each arm, and chooses with a softmax over a weighted sum of value, habit and a
preference for the plus ports. After each outcome:

- the chosen arm's value moves toward the reward, by different amounts for better-than-expected and
  worse-than-expected outcomes (asymmetric learning rates);
- the unchosen arms' values fade toward zero (forgetting);
- the habit trace grows for the chosen arm and fades for the others.

Values carry over across sessions; the habit trace resets at each session start. Simpler "ablated"
versions switch individual parts off, to test which parts the mice need.

## src/agents/kernel_q_agent.py::fit

Fits a version of the model to one mouse by maximum likelihood, from several starting points within
allowed ranges, keeping the best.

## src/agents/kernel_q_agent.py::log_likelihood

How probable the mouse's choices are under given parameters (and how that changes with each
parameter, which speeds up fitting).

## src/agents/kernel_q_agent.py::trial_log_probabilities

The probability the model gave to the mouse's choice on every trial, so any subset of trials (for
example a held-out problem) can be scored.

## src/agents/kernel_q_agent.py::parameters_at_bounds

Lists fitted parameters that ended up at the edge of their allowed range, which means the best value
may lie outside it; read those with care.

## src/agents/mouse_agents.py

The original simple simulated mice: a template for an agent and a basic Q-learner (softmax over
values with a bonus for repeating the last arm), used with the simple drifting-task simulator.

## src/agents/mouse_agents.py::QLearningMouse

A basic Q-learner that chooses by softmax over its values plus a stickiness bonus, and updates the
chosen arm's value toward the reward.

## src/agents/neural_bandit.py

Small neural-network models of choice, matched to a published code release (hybrid RNNs for reward
learning). Instead of hand-written rules, these networks learn from data how to turn past choices
and rewards into choice preferences. Comparing them with the simple models shows how much behavior
the simple models miss.

## src/agents/neural_bandit.py::trace_sequence

Runs a fitted network through one session and returns, trial by trial, its choice probabilities and
internal values.

## src/agents/neural_bandit.py::probe_reward_module

Feeds the network's reward module a grid of prior values and rewards to see how it updates an arm's
value: a way to look inside the network.

## src/agents/null_agents.py

Negative controls: agents that never look at what the other arms are worth. Each decides in two
steps. First, stay or switch, based only on what the current arm just paid. Second, if switching,
which of the two other arms to go to: at random, or by how recently each was chosen, never by value.
Variants: plain WSLS (only "rewarded or not" matters), WSLS (Recency), WSLS (Magnitude) (staying
grows with payout size), WSLS (Graded) and WSLS (Habit) (staying also grows with habit). Because
none of them can use the alternatives' values, any "value effect" an analysis finds in their
replays is produced by the analysis or the schedule, not by value-guided choice. That is what makes
them the right nulls.

## src/agents/null_agents.py::decisions

Turns one mouse's trials into stay/switch decisions (consecutive trial pairs within a session), the
form the null agents are fitted to.

## src/agents/null_agents.py::fit_null_agents

Fits one kind of null agent to every mouse by maximum likelihood.

## src/agents/null_agents.py::log_likelihood

How probable a mouse's choices are under a null agent.

## src/agents/null_agents.py::NullPolicy

A null agent as a simulated mouse that can play the task, for making replays.

## src/agents/paper_rl_fitting.py

Fits the RL and neural models the way the published paper did: split the data into training,
validation and test sets (by mouse or by problem), train with the AdamW optimizer, choose settings
on the validation set only, and score on the test set at the very end. Runs are saved and can be
resumed.

## src/agents/paper_rl_fitting.py::split_by_mouse

Splits mice into training, validation and test groups so no mouse's data appear in more than one.

## src/agents/paper_rl_fitting.py::split_by_problem

Splits by problem instead: the same problem number is held out for every mouse.

## src/agents/paper_rl_fitting.py::run_paper_sweep

Runs (or resumes) the training sweep over model settings, picks the best on validation data, and
scores the test data only once everything has finished.

## src/agents/paper_rl_fitting.py::run_problem_sweep

The same sweep with leave-one-problem-out folds, choosing settings separately in each fold.

## src/agents/parameter_recovery.py

Can the fitting recover the parameters that generated the data? Simulates many sessions with known
parameters under different conditions, fits the model to each, and compares the estimates with the
truth. Large errors would mean the fitted parameters of real mice cannot be trusted.

## src/agents/parameter_recovery.py::recovery_design

The grid of simulation conditions (parameter values, session lengths, repeats), each with its own
reproducible random seed.

## src/agents/parameter_recovery.py::run_recovery

Simulates and fits every condition, saving each result so the sweep can be resumed.

## src/agents/parameter_recovery.py::summarize_recovery

Per condition and parameter, how far estimates are from the truth and how many fits failed.

## src/agents/rank_agent.py

A toy agent that follows the ranked value rule exactly, used to show that the rank tests detect it.
It remembers each arm's last payout, sits on the arm it believes is best, and moves to the best
alternative the moment the current arm's payout drops below it. It never compares with the worst
arm, so the worst arm is only chosen once the arm above it falls below it. A small lapse rate makes
it occasionally choose at random.

## src/agents/released_rl.py

The five-parameter RL model exactly as configured in the paper's released code, kept beside the
six-parameter version described in the paper (`best_rl.py`) because the two differ.

## src/agents/run_store.py

Shared machinery for long runs that can be resumed: each run's folder is named by a fingerprint of
everything that determines its results, so an identical rerun finds its saved work, and files are
written so that an interrupted write never leaves a broken file.

## src/agents/trajectory_viewer.py

An interactive notebook viewer: pick a mouse and a held-out session and see how each neural model's
predictions and internal values evolve trial by trial.

## src/agents/trajectory_viewer.py::trajectory_viewer

Builds the dropdowns and plot for the viewer.

## src/agents/value_agent.py

The basic four-parameter value learner. It keeps a value for each arm, moves the chosen arm's value
toward each reward by a fixed fraction (the learning rate), and chooses with a softmax over values
plus a bonus for repeating the last choice and a preference for plus ports. Values start at zero in
each problem and carry over between its sessions.

## src/agents/value_agent.py::fit

Fits the four parameters to one mouse by maximum likelihood from several starting points.

## src/agents/value_agent.py::log_likelihood

How probable a mouse's choices are under given parameters.

## src/agents/value_agent.py::trial_log_probabilities

The probability the model gave each of the mouse's choices, so any subset of trials can be scored.

## src/agents/value_agent.py::parameters_at_bounds

Lists fitted parameters at the edge of their allowed range.

## src/funcimg_analysis/background.py

In two-photon imaging, each cell is a small region of interest (ROI) whose brightness over time is
its fluorescence trace F. Some signal is common to the whole field of view (brain motion, blood
flow, light leaking in), and it can make every cell look as if it responds. This file treats all
pixels outside the ROIs as one big "background" ROI and computes its trace, so the shared part can
be measured and removed.

## src/funcimg_analysis/background_io.py

Reads the large registered movie files once, in chunks, to compute the background traces of both
channels, and caches the result so later runs skip the slow read.

## src/funcimg_analysis/background_io.py::load_background

The background trace of the green and red channels of one plane, plus the traces of two
non-overlapping halves of the background (to measure how reliable it is) and the images needed for a
sanity figure.

## src/funcimg_analysis/collect.py

Runs an analysis on every imaging run of one processing variant and pools the results. A run is
only used if it has both extracted cells and frame times on the behavior clock; any run left out is
listed with the reason, never dropped silently.

## src/funcimg_analysis/collect.py::collect_pulse_responses

Pools every cell's reward response, split by the number of reward pulses, across all usable runs.

## src/funcimg_analysis/collect.py::collect_background_responses

The same reward response computed on each run's background trace, treated exactly like a cell. If
the background responds like the cells, part of the cells' "response" may be field-wide signal
rather than the cells' own activity.

## src/funcimg_analysis/collect.py::top_cells

The best-ranked cells of a run (by a chosen score), with what is needed to plot them.

## src/funcimg_analysis/collect_flash.py

Reads only the frames of the movie needed for the flash check (see `flash_check.py`) and a short
clip, with the results cached so the movie need not be read over the network again.

## src/funcimg_analysis/collect_motion.py

Runs the motion check (see `motion_check.py`) on every run.

## src/funcimg_analysis/collect_residual.py

Computes background-corrected ΔF/F0 (see `residual_dff.py`) for every run and lines it up with the
trial outcomes, next to the uncorrected version, so the two can be compared.

## src/funcimg_analysis/collect_residual.py::collect_residual_dff

For every run: corrected and uncorrected ΔF/F0 of each cell, aligned to the outcomes, plus which
cells were excluded and why.

## src/funcimg_analysis/collect_residual.py::summary_table

One row per run and channel: how many cells, how many excluded, and how strongly cells were coupled
to the background.

## src/funcimg_analysis/collect_reward_cells.py

Finds the reward-responsive cells of every run (see `reward_cells.py`), for both the uncorrected and
the background-corrected traces.

## src/funcimg_analysis/collect_shared.py

Runs the shared-signal analysis (see `shared_signal_run.py`) on every run: do cells share signal
beyond the background, and does the reward response survive removing the background?

## src/funcimg_analysis/collect_shared.py::collect_shared_signal

Per run and channel: correlations of cells with the background and with each other, the same after
removing the background, chance levels from time-shifted traces, and the reward responses before and
after removing the background.

## src/funcimg_analysis/event_frames.py

Imaging frames arrive about every 133 ms, much slower than behavior events. This file assigns each
event (for example a reward) to its nearest imaging frame and cuts windows of whole frames around it.
No interpolation is used, so what you see is what the microscope recorded.

## src/funcimg_analysis/event_frames.py::pooled_mean_sem

The mean response and its standard error across every cell of every run, per category (for example
per number of reward pulses). Each cell counts once.

## src/funcimg_analysis/event_frames.py::top_cells_by_score

The cells with the highest score (for example a response index), for plotting examples.

## src/funcimg_analysis/flash_check.py

A visual check that the most active-looking cells really are cells lighting up. A real calcium
transient brightens the cell's own pixels in the green channel and stops at its edge, and does
nothing in the red channel (which has no calcium indicator). Motion instead shows as bands at edges
in both channels, and a field-wide signal brightens everything. This file makes the images and
traces around each cell's biggest events so you can tell which it is.

## src/funcimg_analysis/flash_check.py::flash_check

For the cells whose traces are most skewed (most transient-like), the images at their peaks and the
traces around them, in both channels.

## src/funcimg_analysis/frame_times.py

Puts every imaging frame on the behavior clock (pyControl time), using the shared sync pulses, and
saves it beside each run. Without this, you cannot know which frame a reward happened in. It also
does the same for the behavior video, so imaging and video frames can be matched. It refuses to
write anything if it cannot show that the traces, frame clock and behavior file come from the same
recording.

## src/funcimg_analysis/frame_times.py::write_frame_times

Computes the behavior-clock time of every frame of a run and saves it as `frame_times.npz`.

## src/funcimg_analysis/frame_times.py::load_frame_times

Reads a run's saved frame times.

## src/funcimg_analysis/motion_check.py

The red channel has no calcium indicator, so any signal all cells share in red is an artifact
(motion, blood flow). This file puts that shared red signal next to the registration's motion
estimates, suggests where an early artifact period ends, and re-checks for real calcium transients
using only the frames after it.

## src/funcimg_analysis/motion_check.py::motion_check

Returns the motion traces, the cells' shared signal per channel, the suggested cut point, and the
transient check with and without the early frames.

## src/funcimg_analysis/residual_dff.py

ΔF/F0 is a cell's change in brightness relative to its baseline brightness F0, the standard measure
of activity. This file removes the part of each cell's fluctuation it shares with the background.
For each cell it fits how strongly the cell follows the background (fitted on slowly-detrended
traces, because cells and background fade at different rates over a session), then subtracts that
much of the background's fluctuation. The cell keeps its own mean brightness, so its F0 stays
meaningful.

## src/funcimg_analysis/residual_dff.py::residual_dff_channel

Background-corrected ΔF/F0 for every cell of one channel, with each cell's coupling to the
background and the cells excluded because their baseline was unusable.

## src/funcimg_analysis/reward_cells.py

Which cells respond to the reward? A cell's response index is how much its activity after the
outcome differs from before it. The trap: if you rank cells by this index and then plot the
top-ranked ones on the same trials, they will look responsive even if the data are pure noise,
because you picked the luckiest ones. So the trials are split in half: cells are ranked on one half
and shown on the other. A null (shifting each cell's trace in time so it no longer lines up with the
outcomes) says how many cells would pass by chance.

## src/funcimg_analysis/reward_cells.py::trial_responses

For each cell and trial: mean activity in the response window minus mean activity in the baseline
window.

## src/funcimg_analysis/reward_cells.py::analyse_reward_cells

Ranks a run's cells by reward response on half the trials, plots them on the other half, and counts
how many beat their own chance level.

## src/funcimg_analysis/run_responses.py

The basic imaging result: each cell's average activity around the outcome of a trial, split by how
many reward pulses were delivered (0–4). Outcome times are the same as in the photometry analysis.
Four views per cell: raw F and ΔF/F0, for the green (activity) and red (structural, no indicator)
channels. Red serves as a control: a "response" in red points to motion or other artifacts.

## src/funcimg_analysis/run_responses.py::pulse_responses

Each cell's mean window around the outcome for each pulse count, in every view, with standard errors.

## src/funcimg_analysis/run_responses.py::plane_views

Raw F and ΔF/F0 of both channels for every cell of a plane.

## src/funcimg_analysis/run_statistics.py

All the shared-signal numbers for one run, channel and measure, and the row of the printed summary
table.

## src/funcimg_analysis/shared_cache.py

A disposable cache for slow shared-signal results, keyed on everything they depend on (trace
contents, settings and code), so a changed input never returns an old answer.

## src/funcimg_analysis/shared_signal.py

Building blocks for the question "do cells share signal beyond the field-wide background?" Traces
are first detrended and put on a common scale so slow drifts do not create fake correlations. The
fair comparison is one cell against the average of the *other* cells, versus the same cell against
the background: both compare one noisy cell with a cleaner average.

## src/funcimg_analysis/shared_signal_null.py

Chance levels by circular shifts: one trace is slid in time relative to the others (wrapping around
the end), which keeps its own structure but breaks its alignment with them. Recomputing the
statistics on shifted traces shows what you would get with no real relation.

## src/funcimg_analysis/shared_signal_run.py

Runs the shared-signal analysis on one imaging run: the statistics, and the reward responses of the
cells before and after subtracting the background's contribution. Only the background differs
between the two, so any change in the response is due to it.

## src/funcimg_analysis/shared_signal_run.py::analyse_shared_signal

Returns the shared-signal statistics and the original and background-removed reward responses of
one run.

## src/funcimg_analysis/transients.py

Do cells have their own calcium transients? A calcium transient rises fast and decays slowly, so a
cell with real transients has a lopsided (positively skewed) trace with more big upward excursions
than downward ones; noise is symmetric. The red channel, which has no indicator, is the reference
for what noise and artifacts alone produce. Green cells clearly more skewed than red suggest real
activity.

## src/funcimg_analysis/transients.py::run_transients

Skewness and excursion counts for every cell in both channels of a run.

## src/funcimg_analysis/transients.py::channel_comparison

Green against the red reference: median skewness, how many green cells exceed red, and event rates.

## src/funcimg_analysis/zong2022_io.py

File reading and writing for a reproduction of the Zong et al. 2022 imaging pipeline. Results go to
a cache in the repo, and only on request to a folder beside the Suite2p output, never inside it.

## src/funcimg_analysis/zong2022_motion.py

Motion-correction quality control as in Zong et al. 2022: finds the main patterns of variation in
the registered movie and checks how much shift is left between frames at either end of each pattern.
Large leftover shifts mean motion correction missed something; changes in the cell layout suggest
the focal plane drifted (z drift).

## src/funcimg_analysis/zong2022_motion.py::nonrigid_motion

Per frame, how differently separate parts of the image moved: the largest distance between the
shifts of any two registration blocks. Large values mean the tissue warped, not just shifted.

## src/funcimg_analysis/zong2022_responses.py

Which cells respond to the outcome, using the Zong et al. 2022 test: a score comparing activity
around the outcome with activity at other times, ranked against the scores of the same trace shifted
in time. A cell counts as responsive if its real score is above the 95th percentile of its shifted
scores.

## src/funcimg_analysis/zong2022_run.py

Runs one imaging run (or every run of a variant) through the whole Zong et al. 2022 pipeline:
signal processing, response test and motion checks, with the slow steps cached.

## src/funcimg_analysis/zong2022_run.py::analyse_runs

Runs the pipeline on every run, listing the runs that could not be analyzed with the reason.

## src/funcimg_analysis/zong2022_run.py::top_responsive_cells

The highest-scoring cells that pass the signal-to-noise threshold and respond.

## src/funcimg_analysis/zong2022_run.py::pool_reward_runs

Mean and standard error across cells of all runs of the response around the outcome, per pulse
count.

## src/funcimg_analysis/zong2022_signal.py

The signal processing of Zong et al. 2022, step by step: subtract part of the surrounding neuropil
signal from each cell; estimate a slowly varying baseline; compute ΔF/F; mark significant transients
(activity above two local standard deviations for long enough); estimate the underlying spikes
("events") with deconvolution; and measure signal-to-noise so noisy cells can be excluded.

## src/funcimg_analysis/zong2022_signal.py::event_counts_by_tau

How many deconvolved events each cell has inside its significant transients, for several assumed
calcium decay times. The decay time is uncertain for this indicator, so this shows how much the
answer depends on it.

## src/photometry_analysis/event_windows.py

Fiber photometry records a brightness signal from a population of neurons through an implanted
fiber. This file cuts windows of that signal around events (for example rewards), on the behavior
clock. A window is used only if the clock covers all of it without gaps, so nothing is padded or
interpolated across missing time.

## src/photometry_analysis/event_windows.py::extract_event_windows

Resamples the signal onto a common time grid around each event that has a complete window, and says
which events were kept.

## src/photometry_analysis/event_windows.py::sync_coverage

The span of time the sync pulses cover: only events inside it can be aligned.

## src/photometry_analysis/pulse_groups.py

Splits outcome-aligned photometry windows by the number of reward pulses (0 means unrewarded), and
tests whether the response grows with reward size.

## src/photometry_analysis/pulse_groups.py::split_by_pulses

Groups the trial windows by pulse count.

## src/photometry_analysis/pulse_groups.py::pulse_response_correlation

How strongly the response grows with pulse count, as a rank correlation (Spearman) over rewarded
trials. Unrewarded trials are left out, because they are aligned to a different event (the no-reward
cue).
