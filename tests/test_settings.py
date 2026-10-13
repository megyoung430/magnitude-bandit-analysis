import pytest
from pydantic import ValidationError

from src.settings import Settings


def test_data_root_read_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BANDIT_DATA_ROOT", str(tmp_path))
    assert Settings(_env_file=None).data_root == tmp_path


def test_data_root_read_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BANDIT_DATA_ROOT", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"BANDIT_DATA_ROOT={tmp_path}\n")
    assert Settings(_env_file=env_file).data_root == tmp_path


def test_missing_data_root_fails_loudly(monkeypatch):
    monkeypatch.delenv("BANDIT_DATA_ROOT", raising=False)
    with pytest.raises(ValidationError, match="data_root"):
        Settings(_env_file=None)


def test_unmounted_data_root_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("BANDIT_DATA_ROOT", str(tmp_path / "not-mounted"))
    with pytest.raises(ValidationError, match="data_root"):
        Settings(_env_file=None)
