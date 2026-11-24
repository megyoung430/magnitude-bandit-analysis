"""Where a run variant's registered plane is, as mini2p laid it out.

Runs preprocessed before mini2p's ``aligned_<channel>/`` folders keep their plane at
``<variant>/suite2p/plane0``. Newer ones are in ``run-XX/<distortion>/<destriping>/`` with
the registration in ``aligned_<channel>/<save_folder>/plane0``, the channel and
``save_folder`` coming from the settings the run was made with. mini2p records those inside
the variant folder (``pipeline_settings.json``), so the plane is found the same way on any
machine the derivatives are mounted on, using mini2p's own functions to read the record and
build the path.
"""

from pathlib import Path

LEGACY_PLANE = Path("suite2p") / "plane0"


def registered_plane(variant_dir) -> Path | None:
    """The plane folder (holding ``ops.npy``) of *variant_dir*, or None if it has none.

    An older run's ``suite2p/plane0`` is used when it exists. Otherwise the run's recorded
    settings give ``[registration] motion_alignment_source`` and ``save_folder``; a run
    without a record is looked for with mini2p's defaults.

    Raises:
        ValueError: If the alignment is "auto" and both channels were registered (mini2p's
            :func:`find_registered_plane`), or the settings record cannot be read.
    """
    from mini2p_preprocessing.config import (
        SETTINGS_RECORD,
        find_registered_plane,
        settings_from_record,
    )

    variant_dir = Path(variant_dir)
    legacy = variant_dir / LEGACY_PLANE
    if (legacy / "ops.npy").is_file():
        return legacy
    record = variant_dir / SETTINGS_RECORD
    if not record.is_file():
        return find_registered_plane(variant_dir)
    registration = settings_from_record(record.read_text()).registration
    return find_registered_plane(
        variant_dir, registration.motion_alignment_source, registration.save_folder
    )


def variant_of(plane_dir, run_dir) -> str:
    """The variant of *plane_dir* as a path relative to *run_dir*, e.g.
    ``with_distortion_correction/with_destriping`` (or ``with_destriping`` for an older run).

    Raises:
        ValueError: If *plane_dir* is not inside *run_dir* or is not a plane folder.
    """
    parts = Path(plane_dir).relative_to(run_dir).parts
    if len(parts) >= 4 and parts[-1] == "plane0" and parts[-3].startswith("aligned_"):
        return Path(*parts[:-3]).as_posix()
    if len(parts) >= 3 and parts[-2:] == LEGACY_PLANE.parts:
        return Path(*parts[:-2]).as_posix()
    raise ValueError(f"{plane_dir} is not a plane folder mini2p writes inside {run_dir}")
