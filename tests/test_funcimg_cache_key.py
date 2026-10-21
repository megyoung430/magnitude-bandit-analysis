"""What the cached shared-signal and background-corrected results are keyed on."""

from pathlib import Path

import pytest

from src.funcimg_analysis import shared_cache


def test_every_module_whose_output_is_cached_is_in_the_code_version():
    sources = {name for name in shared_cache.CODE_FILES}
    assert {
        "collect_residual.py",
        "residual_dff.py",
        "run_responses.py",
        "shared_signal_run.py",
    } <= sources
    folder = Path(shared_cache.__file__).parent
    assert all((folder / name).is_file() for name in sources)


def test_the_code_version_depends_on_the_installed_mini2p():
    assert shared_cache.code_version("mini2p 1") != shared_cache.code_version("mini2p 2")
    assert shared_cache.code_version("same") == shared_cache.code_version("same")


def test_the_installed_mini2p_is_identified_by_version_and_install_source():
    pytest.importorskip("mini2p_preprocessing")
    identity = shared_cache.mini2p_identity()
    assert identity and "not installed" not in identity
