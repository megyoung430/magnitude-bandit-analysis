"""Behaviour of the shared run-store helpers, on synthetic inputs only."""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from src.agents import run_store
from src.agents.run_store import (
    ProvenanceMismatchWarning,
    at_bound,
    fingerprint_sequences,
    identity_hash,
    library_versions,
    manifest_dir,
    write_atomically,
    write_json_atomically,
)


def make_sequences(n=2):
    return [
        dict(
            mouse=f"m{i}",
            problem=5 + i,
            session=f"ses-{i}",
            towers=["A", "B", "C"],
            actions=np.arange(4) % 3,
            rewards=np.arange(4, dtype=float),
        )
        for i in range(n)
    ]


class TestWriteAtomically:
    def test_writes_the_bytes_and_leaves_no_temporary_file(self, tmp_path):
        path = tmp_path / "out.bin"
        write_atomically(path, lambda handle: handle.write(b"abc"))
        assert path.read_bytes() == b"abc"
        assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]

    def test_failed_write_keeps_the_old_file(self, tmp_path):
        path = tmp_path / "out.bin"
        path.write_bytes(b"old")

        def explode(handle):
            handle.write(b"partial")
            raise RuntimeError("interrupted")

        with pytest.raises(RuntimeError):
            write_atomically(path, explode)
        assert path.read_bytes() == b"old"

    def test_failed_write_removes_the_temporary_file(self, tmp_path):
        def explode(handle):
            handle.write(b"partial")
            raise RuntimeError("interrupted")

        with pytest.raises(RuntimeError):
            write_atomically(tmp_path / "out.bin", explode)
        assert list(tmp_path.iterdir()) == []

    def test_same_stem_files_do_not_share_a_temporary_file(self, tmp_path):
        seen = []

        def record(handle):
            seen.append(handle.name)
            handle.write(b"x")

        write_atomically(tmp_path / "x.json", record)
        write_atomically(tmp_path / "x.npz", record)
        assert len(set(seen)) == 2

    def test_data_is_synced_before_the_rename(self, tmp_path, monkeypatch):
        events = []
        real_fsync, real_replace = os.fsync, os.replace
        monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
        monkeypatch.setattr(
            run_store.Path,
            "replace",
            lambda self, target: (events.append("replace"), real_replace(self, target))[1],
        )
        write_atomically(tmp_path / "out.bin", lambda handle: handle.write(b"abc"))
        assert events == ["fsync", "replace"]

    def test_json_matches_json_dumps_and_rejects_nan(self, tmp_path):
        path = tmp_path / "out.json"
        write_json_atomically(path, {"b": 1, "a": [1.5, None]}, indent=2)
        assert path.read_text() == json.dumps({"b": 1, "a": [1.5, None]}, indent=2)
        with pytest.raises(ValueError):
            write_json_atomically(path, {"x": float("nan")}, indent=2)


class TestIdentityHash:
    def test_is_the_first_twelve_digits_of_the_sha256_of_sorted_json(self):
        text = json.dumps({"a": [1, 2], "b": 2}, sort_keys=True, indent=2)
        assert (
            identity_hash({"b": 2, "a": [1, 2]}) == hashlib.sha256(text.encode()).hexdigest()[:12]
        )

    def test_equal_numbers_hash_equally(self):
        assert identity_hash({"x": 1, "y": [2.0]}) == identity_hash({"x": 1.0, "y": (2,)})
        assert identity_hash({"x": np.float64(0.5), "n": np.int64(3)}) == identity_hash(
            {"x": 0.5, "n": 3.0}
        )

    def test_different_numbers_and_booleans_hash_differently(self):
        assert identity_hash({"x": 0.5}) != identity_hash({"x": 0.25})
        assert identity_hash({"x": True}) != identity_hash({"x": 1})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), np.float64("nan")])
    def test_non_finite_numbers_are_rejected(self, bad):
        with pytest.raises(ValueError, match="non-finite"):
            identity_hash({"nested": [{"x": bad}]})


class TestManifestDir:
    def test_named_by_the_identity_hash_and_holds_the_manifest(self, tmp_path):
        identity = {"b": 2, "a": [1, 2]}
        output = manifest_dir(tmp_path / "new" / "root", identity)
        assert output.name == identity_hash(identity)
        assert (output / "manifest.json").read_text() == json.dumps(
            identity, sort_keys=True, indent=2
        )
        assert manifest_dir(tmp_path / "new" / "root", identity) == output

    def test_manifest_keeps_numbers_as_given(self, tmp_path):
        output = manifest_dir(tmp_path, {"bound": 1.0})
        assert json.loads((output / "manifest.json").read_text()) == {"bound": 1.0}
        assert '"bound": 1.0' in (output / "manifest.json").read_text()

    def test_independent_of_key_order(self, tmp_path):
        first = manifest_dir(tmp_path, {"a": 1, "b": 2})
        assert manifest_dir(tmp_path, {"b": 2, "a": 1}) == first

    def test_different_identities_get_different_directories(self, tmp_path):
        assert manifest_dir(tmp_path, {"v": 1}) != manifest_dir(tmp_path, {"v": 2})

    def test_provenance_may_not_shadow_identity(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate"):
            manifest_dir(tmp_path, {"v": 1}, {"v": 2})
        assert list(tmp_path.iterdir()) == []

    def test_provenance_is_written_once_and_not_overwritten_on_resume(self, tmp_path):
        identity = {"algorithm_version": 1}
        first = manifest_dir(tmp_path, identity, {"source": "text one"})
        with pytest.warns(ProvenanceMismatchWarning):
            second = manifest_dir(tmp_path, identity, {"source": "text two"})
        assert first == second
        manifest = json.loads((second / "manifest.json").read_text())
        assert manifest == {"algorithm_version": 1, "source": "text one"}

    def test_resume_with_changed_source_warns_naming_it_and_the_version(self, tmp_path):
        identity = {"algorithm_version": 1}
        old = {"module": "a", "code": {"fit": "x", "loss": "y"}}
        manifest_dir(tmp_path, identity, old)
        new = {"module": "a", "code": {"fit": "x2", "loss": "y"}}
        with pytest.warns(
            ProvenanceMismatchWarning, match=r"code\.fit.*ALGORITHM_VERSION"
        ) as caught:
            manifest_dir(tmp_path, identity, new)
        message = str(caught[0].message)
        assert "code.loss" not in message and "module" not in message

    def test_every_resume_warns_not_just_the_first(self, tmp_path):
        manifest_dir(tmp_path, {"v": 1}, {"module": "a"})
        for _ in range(3):
            with pytest.warns(ProvenanceMismatchWarning):
                manifest_dir(tmp_path, {"v": 1}, {"module": "b"})

    def test_importing_the_module_registers_an_always_filter(self):
        # pytest resets the warning filters around each test, so check in a fresh interpreter.
        code = (
            "import warnings\n"
            "from src.agents.run_store import ProvenanceMismatchWarning\n"
            "print(any(f[0] == 'always' and f[2] is ProvenanceMismatchWarning "
            "for f in warnings.filters))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.stdout.strip() == "True", result.stderr

    def test_resume_with_unchanged_source_is_silent(self, tmp_path, recwarn):
        provenance = {"module": "a", "code": {"fit": "x"}}
        manifest_dir(tmp_path, {"v": 1}, provenance)
        manifest_dir(tmp_path, {"v": 1}, provenance)
        assert len(recwarn) == 0

    def test_existing_directory_without_a_manifest_gets_one(self, tmp_path):
        (tmp_path / identity_hash({"v": 1})).mkdir()
        output = manifest_dir(tmp_path, {"v": 1}, {"source": "s"})
        assert json.loads((output / "manifest.json").read_text()) == {"v": 1, "source": "s"}


def test_library_versions_reports_installed_distributions():
    assert library_versions("numpy") == {"numpy": np.__version__}
    with pytest.raises(Exception, match="no-such-distribution"):
        library_versions("no-such-distribution")


class TestFingerprintSequences:
    def test_changes_with_rewards_order_and_identity(self):
        data = make_sequences(2)
        base = fingerprint_sequences(data)
        first = fingerprint_sequences(data[:1])
        assert fingerprint_sequences(make_sequences(2)) == base
        assert fingerprint_sequences(data[::-1]) != base
        assert fingerprint_sequences([dict(data[0], rewards=data[0]["rewards"] + 1)]) != first
        assert fingerprint_sequences([dict(data[0], mouse="x")]) != first

    def test_numpy_integer_problem_numbers_hash_like_python_ints(self):
        data = make_sequences(1)
        numpy_problem = [dict(data[0], problem=np.int64(5))]
        assert fingerprint_sequences(numpy_problem) == fingerprint_sequences(data)

    def test_changes_with_towers_actions_and_session(self):
        data = make_sequences(1)
        base = fingerprint_sequences(data)
        assert fingerprint_sequences([dict(data[0], towers=["A", "B", "D"])]) != base
        assert fingerprint_sequences([dict(data[0], actions=data[0]["actions"][::-1])]) != base
        assert fingerprint_sequences([dict(data[0], session="other")]) != base

    def test_non_integral_problem_numbers_fail_loudly(self):
        data = make_sequences(1)
        with pytest.raises(ValueError, match="integral"):
            fingerprint_sequences([dict(data[0], problem=5.5)])
        assert fingerprint_sequences([dict(data[0], problem=5.0)]) == fingerprint_sequences(data)

    def test_empty_input_is_the_empty_sha256(self):
        assert fingerprint_sequences([]).startswith("e3b0c44298fc1c14")


class TestAtBound:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.00105, True), (0.00115, False), (0.5, False), (1.00005, True)],
    )
    def test_tolerance_edges(self, value, expected):
        assert at_bound(value, (0.001, 1.0), atol=1e-4) is expected

    def test_none_limit_is_unbounded_not_an_error(self):
        # best_rl-style bounds leave one side open.
        assert at_bound(1e9, (0.0, None), atol=1e-4) is False
        assert at_bound(0.0, (0.0, None), atol=1e-4) is True
        assert at_bound(5.0, (None, None), atol=1e-4) is False

    def test_tolerance_is_required(self):
        with pytest.raises(TypeError):
            at_bound(1.0, (1.0,))
