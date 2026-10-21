"""A disposable on-disk cache of whole-run shared-signal results.

A result is keyed on a hash of everything it is computed from: the content of every trace, the
frame times and outcomes, every setting, and the source of the analysis code. Deleting the cache
directory only costs time, and an edited trace, setting or piece of code never finds a stale file.
"""

import hashlib
import os
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_SUBDIR = "funcimg_shared"
# every module whose code shapes a cached result (the shared-signal and the background-corrected
# ΔF/F0 results), so that editing one never finds a stale file
CODE_FILES = (
    "shared_signal.py",
    "shared_signal_null.py",
    "run_statistics.py",
    "shared_signal_run.py",
    "event_frames.py",
    "run_responses.py",
    "residual_dff.py",
    "collect_residual.py",
)
MINI2P_DISTRIBUTION = "mini2p-preprocessing"


def mini2p_identity() -> str:
    """The installed mini2p's version and install source (the commit, for a git install).

    ΔF/F0 baselines come from mini2p, so a cached result is only valid for the mini2p that made it.
    """
    from importlib import metadata

    try:
        distribution = metadata.distribution(MINI2P_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return "mini2p not installed"
    source = distribution.read_text("direct_url.json") or ""
    return f"{distribution.version} {source}"


def code_version(mini2p: str) -> str:
    """A hash of the source of :data:`CODE_FILES` and of the *mini2p* identity."""
    folder = Path(__file__).parent
    hasher = hashlib.sha256(b"".join((folder / name).read_bytes() for name in CODE_FILES))
    hasher.update(mini2p.encode())
    return hasher.hexdigest()


CODE_VERSION = code_version(mini2p_identity())


def _feed(hasher, value) -> None:
    """Add *value* to *hasher*: arrays by dtype, shape and bytes, containers by their items."""
    if isinstance(value, np.ndarray):
        hasher.update(f"array{value.dtype}{value.shape}".encode())
        hasher.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, pd.DataFrame):
        _feed(hasher, value.to_csv())
    elif isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value, key=repr):
            _feed(hasher, key)
            _feed(hasher, value[key])
    elif isinstance(value, list | tuple):
        hasher.update(f"{type(value).__name__}{len(value)}".encode())
        for item in value:
            _feed(hasher, item)
    else:
        hasher.update(repr(value).encode())


def content_key(parts: dict) -> str:
    """A hash of *parts* and the code version; dict order does not matter, array content does."""
    hasher = hashlib.sha256()
    _feed(hasher, {"code": CODE_VERSION, **parts})
    return hasher.hexdigest()[:32]


def cached_result(cache_dir, key: str, compute: Callable[[], dict]) -> dict:
    """The result saved under *key*, or ``compute()`` saved there first on a miss.

    A file that cannot be read (truncated by an interrupted run) is recomputed and replaced,
    which is safe because the cache is disposable.
    """
    cache_file = Path(cache_dir) / f"{key}.pkl"
    if cache_file.is_file():
        try:
            with open(cache_file, "rb") as handle:
                return pickle.load(handle)
        except (EOFError, pickle.UnpicklingError):
            pass
    result = compute()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")  # only complete files are read
    with open(tmp, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)
    return result
