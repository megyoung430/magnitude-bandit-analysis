import json
from pathlib import Path

import pytest

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from mini2p_preprocessing.config import SETTINGS_RECORD, settings_from_toml

from src.funcimg_import.planes import registered_plane, variant_of

VARIANT = "with_distortion_correction/with_destriping"


def write_settings(variant_dir: Path, save_folder="suite2p_rigid_sigma3", source="structural"):
    """The pipeline_settings.json mini2p writes in each variant folder."""
    settings = settings_from_toml(
        {
            "distortion": {"mode": "off"},
            "registration": {"save_folder": save_folder, "motion_alignment_source": source},
        }
    )
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / SETTINGS_RECORD).write_text(settings.model_dump_json())


def make_plane(variant_dir: Path, *parts) -> Path:
    plane = variant_dir.joinpath(*parts)
    plane.mkdir(parents=True)
    (plane / "ops.npy").write_bytes(b"")
    return plane


def batch_manifest(variant_dir: Path, config="/ceph/elsewhere/batch_config_x.toml"):
    """A source_tiff.json as mini2p's batch writes it, naming a config on another mount."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_tiff": "/raw/71rtask_00039.tif", "pipeline_config": config}
    (variant_dir / "source_tiff.json").write_text(json.dumps(manifest))


def test_the_plane_follows_the_settings_the_run_recorded(tmp_path):
    variant_dir = tmp_path / "run-01" / VARIANT
    write_settings(variant_dir)
    batch_manifest(variant_dir)  # a config path this machine cannot see is not needed
    make_plane(variant_dir, "aligned_structural", "suite2p", "plane0")  # another registration
    wanted = make_plane(variant_dir, "aligned_structural", "suite2p_rigid_sigma3", "plane0")
    assert registered_plane(variant_dir) == wanted


def test_the_recorded_alignment_channel_is_used(tmp_path):
    variant_dir = tmp_path / "run-01" / VARIANT
    write_settings(variant_dir, "suite2p", "functional")
    make_plane(variant_dir, "aligned_structural", "suite2p", "plane0")
    wanted = make_plane(variant_dir, "aligned_functional", "suite2p", "plane0")
    assert registered_plane(variant_dir) == wanted


def test_a_recorded_registration_that_is_not_there_gives_none(tmp_path):
    variant_dir = tmp_path / "run-01" / VARIANT
    write_settings(variant_dir)
    make_plane(variant_dir, "aligned_structural", "suite2p", "plane0")
    assert registered_plane(variant_dir) is None


def test_an_older_run_keeps_its_suite2p_plane_whatever_it_recorded(tmp_path):
    variant_dir = tmp_path / "run-01" / "with_destriping"
    batch_manifest(variant_dir)
    (variant_dir / SETTINGS_RECORD).write_text('{"phase": {"method": "chunked"}}')  # unreadable
    wanted = make_plane(variant_dir, "suite2p", "plane0")
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
        (f"{VARIANT}/aligned_structural/suite2p/plane0", VARIANT),  # mini2p's default folder
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
