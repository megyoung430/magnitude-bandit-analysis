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
    """Machine-specific settings; each field is read from ``BANDIT_<FIELD>`` or ``.env``.

    Attributes:
        data_root (DirectoryPath): Directory holding the experiment folders (set with
            ``BANDIT_DATA_ROOT``). Required, and it must exist, so an unmounted share fails
            at load time.
        cache_dir (Path): Disposable on-disk cache for slow steps such as importing raw
            sessions (``BANDIT_CACHE_DIR``). Default ``<repo>/.cache``.
    """

    model_config = SettingsConfigDict(
        env_prefix="BANDIT_",
        # anchored to the repo so scripts and notebooks find it from any cwd
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )

    # DirectoryPath must exist: an unmounted share fails here, not deep in import_data
    data_root: DirectoryPath
    # disposable on-disk cache for slow steps such as importing raw sessions
    cache_dir: Path = REPO_ROOT / ".cache"


@lru_cache
def get_settings() -> Settings:
    """Load and validate settings once per process.

    Returns:
        Settings: The cached settings object.

    Raises:
        pydantic.ValidationError: If ``BANDIT_DATA_ROOT`` is unset or not an existing directory.
    """
    return Settings()
