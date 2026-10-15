"""Shared plumbing for resumable runs: atomic writes, manifest directories, fingerprints.

A resumable run names its output directory by hashing a description of everything that
determines its results (the *identity*), so an identical rerun finds its saved work. The
identity holds an explicit algorithm version and the configuration, never source text:
editing a comment or refactoring must not orphan hours of finished fits. Source text can
still be recorded as *provenance*, which is written to the manifest but kept out of the hash.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

import numpy as np


def write_atomically(path: Path, write: Callable[[IO[bytes]], None]) -> None:
    """Call ``write(handle)`` on a temporary file, then move it over *path*.

    An interrupted run leaves either the old file or the new one, never a partial file.
    """
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        write(handle)
    temporary.replace(path)


def write_json_atomically(path: Path, payload, *, indent: int) -> None:
    """Atomically write *payload* as JSON; NaN and infinity raise rather than becoming null."""
    text = json.dumps(payload, indent=indent, allow_nan=False)
    write_atomically(path, lambda handle: handle.write(text.encode()))


def manifest_dir(output_root, identity: dict, provenance: dict | None = None) -> Path:
    """Create ``<output_root>/<hash of identity>/`` and write ``manifest.json`` into it.

    Only *identity* is hashed (first 12 hex digits of the SHA-256 of its canonical JSON,
    sorted keys). *provenance* is merged into the manifest file for the reader's benefit
    and does not affect the directory name.
    """
    provenance = {} if provenance is None else provenance
    overlap = identity.keys() & provenance.keys()
    if overlap:
        raise ValueError(f"Provenance keys duplicate identity keys: {sorted(overlap)}")
    identity_text = json.dumps(identity, sort_keys=True, indent=2)
    output = Path(output_root) / hashlib.sha256(identity_text.encode()).hexdigest()[:12]
    output.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps({**identity, **provenance}, sort_keys=True, indent=2)
    write_atomically(output / "manifest.json", lambda handle: handle.write(manifest_text.encode()))
    return output


def fingerprint_sequences(sequences: list[dict]) -> str:
    """SHA-256 over identities, towers, actions and rewards, in input order."""
    fingerprint = hashlib.sha256()
    for s in sequences:
        header = [s["mouse"], int(s["problem"]), s["session"], s["towers"]]
        fingerprint.update(json.dumps(header).encode())
        fingerprint.update(np.asarray(s["actions"], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(s["rewards"], dtype=np.float64).tobytes())
    return fingerprint.hexdigest()


def at_bound(value: float, limits: Sequence[float | None], *, atol: float) -> bool:
    """True if *value* lies within *atol* of a finite limit; a None limit is unbounded."""
    return bool(any(np.isclose(value, b, atol=atol, rtol=0) for b in limits if b is not None))
