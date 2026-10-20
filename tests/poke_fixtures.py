"""Tiny pyControl-like recordings for the poke analysis tests.

:class:`Recording` appends rows in the order a pyControl log has them (``time``, ``type``,
``subtype``, ``content``) and exposes one method per thing the task does, so a test reads as the
mouse's behaviour. The state sequences follow the drifting-value task:

- ``wait_init`` -> (initiating ``B2_in``) -> ``wait_choice``
- in ``wait_choice`` a poke enters ``hold_poke``; leaving early returns to ``wait_choice``, holding
  until ``poke_hold_timer`` goes to ``poked_in``
- a long poke at a choice tower is the choice: a trial print row follows ``poked_in``;
  a long poke elsewhere returns to ``wait_choice`` without a print
- after the choice: ``choice`` (rewarded only), ``ITI``, ``wait_init``
"""

import json

import pandas as pd

TOWERS = ["A1", "B1", "C1"]


class Recording:
    """A growing list of log rows; every method returns ``self`` so calls chain."""

    def __init__(self, *, towers=None, init="B2", start=True):
        self.rows = [
            (0.0, "info", "task_name", "Meg\\3x3_field_drifting_value_bandit"),
            # a session of the importer always has one; the port-finding check reads it
            (0.0, "variable", "run_start", json.dumps({"found_ports": False, "found_t_thresh": 4})),
        ]
        self.t = 0.0
        self.towers = list(towers or TOWERS)
        self.init = init
        self.num_t = 0
        self._trial_long_pokes: list[str] = []
        if start:
            self.state("wait_init")

    def _add(self, type_, subtype, content):
        self.t += 0.01
        self.rows.append((round(self.t, 3), type_, subtype, content))
        return self

    def state(self, name):
        return self._add("state", None, name)

    def poke_in(self, port):
        return self._add("event", "input", f"{port}_in")

    def poke_out(self, port):
        return self._add("event", "input", f"{port}_out")

    def timer(self, name):
        return self._add("event", "timer", name)

    def sync(self):
        return self._add("event", "sync", "rsync")

    def wait_init_poke(self, port):
        """A poke at a port other than the initiation tower while waiting to initiate."""
        return self.poke_in(port).poke_out(port)

    def initiate(self):
        """The initiating poke: ``init_in`` then the state change to ``wait_choice``."""
        self._trial_long_pokes = []
        return self.poke_in(self.init).state("wait_choice").poke_out(self.init)

    def short_poke(self, port):
        return self.poke_in(port).state("hold_poke").poke_out(port).state("wait_choice")

    def long_poke(self, port):
        """A held poke that is not the choice: back to ``wait_choice`` afterwards."""
        self._long(port)
        self._trial_long_pokes.append(port)
        return self.state("wait_choice")

    def _long(self, port):
        return self.poke_in(port).state("hold_poke").timer("poke_hold_timer").state("poked_in")

    def choose(self, port, *, rewarded=True, choice_pokes=(), held=None):
        """The choice long poke, the trial print row, and the ``choice`` state when rewarded.

        *choice_pokes* are ``_in`` events logged during the ``choice`` state (drinking at the
        chosen port), which lie before the ``ITI`` window. *held* is the port actually held
        when it differs from the printed chosen *port* (a log inconsistency).
        """
        self.num_t += 1
        self._long(held or port)
        printed = {
            "num_t": self.num_t,
            "num_blocks": 1,
            "chosen_tow": port,
            "chosen_rank": "best",
            "num_long_pokes": len(self._trial_long_pokes),
            "long_pokes": list(self._trial_long_pokes),
            "c_tows": self.towers,
            "curr_rew_mag": {t: 1 for t in self.towers},
            "rank_counts": {"best": self.num_t, "second": 0, "third": 0},
        }
        self._add("variable", "print", json.dumps(printed))
        if rewarded:
            self.state("choice")
            for p in choice_pokes:
                self.poke_in(p)
        return self

    def port_finding_trial(
        self, port="A1", *, pre_init=(), long_before=(), after_choice=(), held=None, chosen=True
    ):
        """One port-finding trial, shaped like a task trial: initiate, long pokes, choice, ITI.

        Port-finding prints carry ``num_t_found`` and ``found_ports`` false but no ``num_t``.
        *pre_init* are pokes in ``wait_init`` before the initiating poke, *long_before* held
        non-choice pokes, *after_choice* pokes after the ``ITI`` state, *held* the port actually
        held when it differs from the printed *port*, and ``chosen=False`` leaves ``chosen_tow``
        out of the print. The recording is left in ``wait_init``.
        """
        self.num_t_found = getattr(self, "num_t_found", 0) + 1
        for p in pre_init:
            self.wait_init_poke(p)
        self.initiate()
        for p in long_before:
            self.long_poke(p)
        self._long(held or port)
        printed = {
            "num_t_found": self.num_t_found,
            "found_ports": False,
            "long_pokes": list(long_before),
        }
        if chosen:
            printed["chosen_tow"] = port
        self._add("variable", "print", json.dumps(printed))
        self.state("choice").iti()
        for p in after_choice:
            self.poke_in(p).poke_out(p)
        return self.state("wait_init")

    def port_finding_timeout(self):
        """A port-finding trial that times out: initiate, then the ITI with no print."""
        return self.initiate().iti().state("wait_init")

    def raw_print(self, printed: dict):
        """A trial print row with exactly the given fields."""
        return self._add("variable", "print", json.dumps(printed))

    def text_print(self, text="trial done"):
        """A plain-text print row, which parses to an empty trial dict."""
        self.num_t += 1
        return self._add("variable", "print", text)

    def iti(self):
        return self.state("ITI")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["time", "type", "subtype", "content"])


def full_trial(
    rec, *, pre_init=(), shorts=(), long_before=(), choice="A1", rewarded=True, after_choice=()
):
    """One trial: pokes while waiting to initiate, initiate, pokes, choose, then the ITI window.

    Args:
        pre_init: Ports poked (in and out) in ``wait_init`` before the initiating poke; they
            belong to the previous trial's after-choice window.
        shorts, long_before: Short pokes and held non-choice pokes between initiation and choice.
        after_choice: Ports poked (in and out) after the ``ITI`` state, before the recording
            moves to the next ``wait_init``.

    The recording is left in ``wait_init``, ready for the next trial's ``pre_init`` and
    initiating poke, or to end there.
    """
    for p in pre_init:
        rec.wait_init_poke(p)
    rec.initiate()
    for p in shorts:
        rec.short_poke(p)
    for p in long_before:
        rec.long_poke(p)
    rec.choose(choice, rewarded=rewarded)
    rec.iti()
    for p in after_choice:
        rec.poke_in(p).poke_out(p)
    rec.state("wait_init")
    return rec


def session_record(frames, **extra):
    """A minimal importer-style session dict around raw frames."""
    frames = list(frames)
    return {
        "df": frames[0] if len(frames) == 1 else frames,
        "data": frames[0] if len(frames) == 1 else frames,
        "choice_towers": set(TOWERS),
        "initiation_tower": "B2",
        **extra,
    }
