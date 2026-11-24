"""Where a run variant's registered plane is, as mini2p laid it out.

mini2p writes each variant of a run to ``run-XX/<distortion>/<destriping>/`` and its
registration to ``aligned_<channel>/<save_folder>/plane0`` inside it, the channel and
``save_folder`` coming from the run's config. Its batch records that config (a frozen copy)
in the variant's ``source_tiff.json``, so the plane is found from the settings the run was
actually made with, using mini2p's own path functions. Runs from before that layout keep
their plane at ``<destriping>/suite2p/plane0``.
"""

import json
from pathlib import Path

BATCH_MANIFEST = "source_tiff.json"
LEGACY_PLANE = Path("suite2p") / "plane0"


def recorded_config(variant_dir) -> Path | None:
    """The config the batch made *variant_dir* with, or None if it recorded none.

    Raises:
        FileNotFoundError: If the recorded config no longer exists, since the plane could
            then only be guessed.
    """
    manifest = Path(variant_dir) / BATCH_MANIFEST
    if not manifest.is_file():
        return None
    recorded = json.loads(manifest.read_text()).get("pipeline_config")
    if not recorded:
        return None
    config = Path(recorded)
    if not config.is_file():
        raise FileNotFoundError(
            f"{manifest} records the config {config}, which is gone; restore it (the batch "
            "writes it to the subject's derivatives folder as batch_config_<time>.toml)."
        )
    return config


def registered_plane(variant_dir) -> Path | None:
    """The plane folder (holding ``ops.npy``) of *variant_dir*, or None if it has none.

    With a recorded config, the alignment channel and ``save_folder`` are that config's
    ``[registration] motion_alignment_source`` and ``save_folder``. Without one, the
    pre-mini2p ``suite2p/plane0`` is used if it exists, and otherwise mini2p's defaults.

    Raises:
        ValueError: If the config's alignment is "auto" and both channels were registered
            (mini2p's :func:`find_registered_plane`).
    """
    from mini2p_preprocessing.config import find_registered_plane, load_settings

    variant_dir = Path(variant_dir)
    config = recorded_config(variant_dir)
    if config is None:
        legacy = variant_dir / LEGACY_PLANE
        if (legacy / "ops.npy").is_file():
            return legacy
        return find_registered_plane(variant_dir)
    registration = load_settings(config).registration
    return find_registered_plane(
        variant_dir, registration.motion_alignment_source, registration.save_folder
    )


def variant_of(plane_dir, run_dir) -> str:
    """The variant of *plane_dir* as a path relative to *run_dir*, e.g.
    ``with_distortion_correction/with_destriping`` (or ``with_destriping`` for an older run).

    Raises:
        ValueError: If *plane_dir* is not inside *run_dir*.
    """
    relative = Path(plane_dir).relative_to(run_dir)
    if relative.parts[-2:] == LEGACY_PLANE.parts:
        return Path(*relative.parts[:-2]).as_posix()
    if len(relative.parts) < 4 or not relative.parts[-3].startswith("aligned_"):
        raise ValueError(f"{plane_dir} is not a plane folder mini2p writes inside {run_dir}")
    return Path(*relative.parts[:-3]).as_posix()
