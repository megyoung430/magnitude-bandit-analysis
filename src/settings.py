"""Machine-specific configuration, read from the environment or a local ``.env``.

Each machine sets ``BANDIT_DATA_ROOT`` to the directory that holds the
experiment folders (e.g. ``3x3_field_magnitude_bandit/``): the SMB mount of
ceph on a laptop, the ceph path on the HPC. See ``.env.example``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BANDIT_",
        # anchored to the repo so scripts and notebooks find it from any cwd
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )

    # DirectoryPath must exist: an unmounted share fails here, not deep in import_data
    data_root: DirectoryPath


@lru_cache
def get_settings() -> Settings:
    """Load and validate settings once per process."""
    return Settings()
