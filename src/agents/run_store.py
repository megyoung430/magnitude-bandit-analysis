"""Shared plumbing for resumable runs: atomic writes, manifest directories, fingerprints.

A resumable run names its output directory by hashing a description of everything that
determines its results (the *identity*), so an identical rerun finds its saved work. The
identity holds explicit algorithm and model versions and the configuration, never this
code's source text: editing a comment or refactoring must not orphan hours of finished
fits. Source text can still be recorded as *provenance*, which is written to the manifest
when the directory is created and kept out of the hash. The functions a caller supplies
(a fit or loss function) are inputs rather than this code, so their source belongs in the
identity.
"""

import hashlib
import json
import math
import os
import warnings
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import IO

import numpy as np


def write_atomically(path: Path, write: Callable[[IO[bytes]], None]) -> None:
    """Call ``write(handle)`` on a temporary file, then move it over *path*.

    The data is flushed to disk before the rename, so an interrupted run (even a power loss) leaves
    either the old file or the new one, never a partial file. The temporary file is named after the
    full file name, so ``x.json`` and ``x.npz`` do not share one, and it is removed if writing
    fails.

    Args:
        path (Path): Destination file. Its directory must exist.
        write (Callable[[IO[bytes]], None]): Called once with the temporary file opened in binary
            write mode; it writes the bytes.
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_atomically(path: Path, payload, *, indent: int) -> None:
    """Atomically write *payload* as JSON; NaN and infinity raise rather than becoming null.

    Args:
        path (Path): Destination file. Its directory must exist.
        payload (object): Anything ``json.dumps`` accepts (dicts, lists, strings, finite numbers,
            bools, None).
        indent (int): Spaces of indentation in the written JSON (keyword-only).

    Raises:
        ValueError: If *payload* contains NaN or infinity.
        TypeError: If *payload* is not JSON serialisable.
    """
    text = json.dumps(payload, indent=indent, allow_nan=False)
    write_atomically(path, lambda handle: handle.write(text.encode()))


class ProvenanceMismatchWarning(UserWarning):
    """A run resumed with source text that differs from the code that started it."""


# Python shows a given warning once per call site by default, which would hide a repeat in a
# long notebook session.
warnings.simplefilter("always", ProvenanceMismatchWarning)


def _canonical(value):
    """*value* with numbers normalised, so equal numbers hash equally and NaN is rejected.

    A float with an integral value becomes an int (``1.0`` and ``1`` are the same
    setting); numpy scalars become Python numbers; tuples become lists.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Identity contains a non-finite number: {value!r}")
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def identity_hash(identity: dict) -> str:
    """First 12 hex digits of the SHA-256 of the canonical JSON of *identity*.

    Keys are sorted and numbers canonicalised, so the hash does not depend on key order or on
    whether a setting was written as ``1`` or ``1.0``.

    Args:
        identity (dict): JSON-compatible description of everything that determines a run's results.
            Numpy scalars and tuples are accepted and normalised.

    Returns:
        str: 12 lowercase hex characters.

    Raises:
        ValueError: If *identity* contains a non-finite number.
    """
    text = json.dumps(_canonical(identity), sort_keys=True, indent=2, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _flatten(provenance: dict, prefix: str = "") -> dict:
    """``{"a": {"b": x}}`` as ``{"a.b": x}``, so a changed entry can be named."""
    flat = {}
    for key, value in provenance.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def changed_sources(stored: dict, current: dict) -> list[str]:
    """Names of the entries of *current* that are new or differ from the *stored* manifest.

    Args:
        stored (dict): Provenance read back from a manifest; nested dicts are compared entry by
            entry.
        current (dict): Provenance of the code now running, same layout as *stored*.

    Returns:
        list[str]: Sorted dotted names (``"a.b"`` for ``{"a": {"b": ...}}``) of entries in *current*
            whose value is missing from or different in *stored*. Entries present only in *stored*
            are not reported.
    """
    old, new = _flatten(stored), _flatten(current)
    return sorted(key for key in new if old.get(key) != new[key])


def manifest_dir(output_root, identity: dict, provenance: dict | None = None) -> Path:
    """Create or reopen ``<output_root>/<identity_hash>/`` and its ``manifest.json``.

    Only *identity* names the directory. *provenance* (source text of the code) is written into the
    manifest when the directory is first created, so the manifest keeps describing the code that
    produced the existing results. On a resume whose current provenance differs from the stored one,
    a warning names the changed entries: the results are reused as they are, so ALGORITHM_VERSION
    must be bumped if the change affects results.

    Args:
        output_root (str | Path): Directory that holds one subdirectory per run identity.
        identity (dict): JSON-compatible description of everything that determines the results;
            hashed to name the directory. See ``identity_hash``.
        provenance (dict | None): Source text of the code, recorded in the manifest but not hashed.
            None means empty. Its keys must not repeat keys of *identity*.

    Returns:
        Path: The run directory, ``output_root / identity_hash(identity)``.

    Raises:
        ValueError: If *provenance* and *identity* share a key, or *identity* has a non-finite
            number.

    Warns:
        ProvenanceMismatchWarning: If the directory exists and *provenance* differs from the stored
            manifest.
    """
    provenance = {} if provenance is None else provenance
    overlap = identity.keys() & provenance.keys()
    if overlap:
        raise ValueError(f"Provenance keys duplicate identity keys: {sorted(overlap)}")
    output = Path(output_root) / identity_hash(identity)
    manifest_file = output / "manifest.json"
    if manifest_file.exists():
        changed = changed_sources(json.loads(manifest_file.read_text()), provenance)
        if changed:
            warnings.warn(
                f"Resuming {output} with source that differs from the code that started it: "
                f"{', '.join(changed)}. Finished work is reused unchanged; bump "
                "ALGORITHM_VERSION (or MODEL_VERSION) if the change affects results.",
                ProvenanceMismatchWarning,
                stacklevel=2,
            )
        return output
    output.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps({**identity, **provenance}, sort_keys=True, indent=2)
    write_atomically(manifest_file, lambda handle: handle.write(manifest_text.encode()))
    return output


def library_versions(*names: str) -> dict[str, str]:
    """Installed version of each named distribution, for a run's identity.

    Args:
        *names: Distribution names (str), e.g. ``"numpy"``.

    Returns:
        dict[str, str]: ``{name: installed version}``.

    Raises:
        importlib.metadata.PackageNotFoundError: If a distribution is not installed.
    """
    return {name: metadata.version(name) for name in names}


def _as_int(value) -> int:
    """*value* as an int; raise if it is not integral rather than truncating it."""
    integer = int(value)
    if integer != value:
        raise ValueError(f"Expected an integral number, got {value!r}")
    return integer


def fingerprint_sequences(sequences: list[dict]) -> str:
    """SHA-256 over identities, towers, actions and rewards, in input order.

    Args:
        sequences (list[dict]): Sessions as built by ``prepare_cohort_sequences``; each needs
            ``mouse``, ``problem`` (integral), ``session``, ``towers``, ``actions`` and ``rewards``.

    Returns:
        str: 64 lowercase hex characters. Changes if any session, trial or order changes.

    Raises:
        ValueError: If a session's ``problem`` is not integral.
    """
    fingerprint = hashlib.sha256()
    for s in sequences:
        header = [s["mouse"], _as_int(s["problem"]), s["session"], s["towers"]]
        fingerprint.update(json.dumps(header).encode())
        fingerprint.update(np.asarray(s["actions"], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(s["rewards"], dtype=np.float64).tobytes())
    return fingerprint.hexdigest()


def at_bound(value: float, limits: Sequence[float | None], *, atol: float) -> bool:
    """True if *value* lies within *atol* of a finite limit; a None limit is unbounded.

    Args:
        value (float): Fitted parameter value.
        limits (Sequence[float | None]): The ``(low, high)`` bounds; None entries are skipped.
        atol (float): Absolute distance at or below which *value* counts as at the bound
            (keyword-only).

    Returns:
        bool: Whether *value* is at any finite limit.
    """
    return bool(any(np.isclose(value, b, atol=atol, rtol=0) for b in limits if b is not None))
