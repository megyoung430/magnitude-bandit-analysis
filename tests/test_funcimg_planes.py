import json
from pathlib import Path

import pytest

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_import.planes import recorded_config, registered_plane, variant_of

VARIANT = "with_distortion_correction/with_destriping"


def write_config(path: Path, save_folder="suite2p_rigid_sigma3", source="structural"):
    path.write_text(
        "[distortion]\n"
        'mode = "off"\n'
        "[registration]\n"
        f'save_folder = "{save_folder}"\n'
        f'motion_alignment_source = "{source}"\n'
    )
    return path


def make_plane(variant_dir: Path, *parts) -> Path:
    plane = variant_dir.joinpath(*parts)
    plane.mkdir(parents=True)
    (plane / "ops.npy").write_bytes(b"")
    return plane


def record(variant_dir: Path, config: Path | None) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_tiff": "/raw/71rtask_00039.tif"}
    if config is not None:
        manifest["pipeline_config"] = str(config)
    (variant_dir / "source_tiff.json").write_text(json.dumps(manifest))


def test_the_plane_follows_the_config_the_batch_recorded(tmp_path):
    variant_dir = tmp_path / "run-01" / VARIANT
    record(variant_dir, write_config(tmp_path / "batch_config.toml"))
    make_plane(variant_dir, "aligned_structural", "suite2p", "plane0")  # another registration
    wanted = make_plane(variant_dir, "aligned_structural", "suite2p_rigid_sigma3", "plane0")
    assert registered_plane(variant_dir) == wanted


def test_the_recorded_alignment_channel_is_used(tmp_path):
    variant_dir = tmp_path / "run-01" / VARIANT
    record(variant_dir, write_config(tmp_path / "c.toml", "suite2p", "functional"))
    make_plane(variant_dir, "aligned_structural", "suite2p", "plane0")
    wanted = make_plane(variant_dir, "aligned_functional", "suite2p", "plane0")
    assert registered_plane(variant_dir) == wanted


def test_a_recorded_registration_that_is_not_there_gives_none(tmp_path):
    variant_dir = tmp_path / "run-01" / VARIANT
    record(variant_dir, write_config(tmp_path / "c.toml"))
    make_plane(variant_dir, "aligned_structural", "suite2p", "plane0")
    assert registered_plane(variant_dir) is None


def test_a_recorded_config_that_is_gone_is_refused(tmp_path):
    variant_dir = tmp_path / "run-01" / VARIANT
    record(variant_dir, tmp_path / "deleted.toml")
    with pytest.raises(FileNotFoundError, match="batch_config"):
        registered_plane(variant_dir)


def test_an_older_run_keeps_its_suite2p_plane(tmp_path):
    variant_dir = tmp_path / "run-01" / "with_destriping"
    record(variant_dir, None)
    wanted = make_plane(variant_dir, "suite2p", "plane0")
    assert recorded_config(variant_dir) is None
    assert registered_plane(variant_dir) == wanted


def test_without_a_record_mini2p_defaults_find_the_plane(tmp_path):
    variant_dir = tmp_path / "run-01" / VARIANT
    wanted = make_plane(variant_dir, "aligned_structural", "suite2p", "plane0")
    assert registered_plane(variant_dir) == wanted


def test_a_variant_without_a_registration_has_no_plane(tmp_path):
    (tmp_path / "run-01" / VARIANT).mkdir(parents=True)
    assert registered_plane(tmp_path / "run-01" / VARIANT) is None


@pytest.mark.parametrize(
    "plane, variant",
    [
        (f"{VARIANT}/aligned_structural/suite2p_rigid/plane0", VARIANT),
        ("with_destriping/suite2p/plane0", "with_destriping"),
    ],
)
def test_the_variant_is_read_back_from_the_plane_path(tmp_path, plane, variant):
    run_dir = tmp_path / "run-01"
    assert variant_of(run_dir / plane, run_dir) == variant


def test_a_folder_that_is_not_a_plane_is_refused(tmp_path):
    run_dir = tmp_path / "run-01"
    with pytest.raises(ValueError, match="not a plane folder"):
        variant_of(run_dir / VARIANT / "plane0", run_dir)
