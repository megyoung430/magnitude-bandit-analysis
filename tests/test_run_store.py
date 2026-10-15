"""Behaviour of the shared run-store helpers, on synthetic inputs only."""

import hashlib
import json

import numpy as np
import pytest

from src.agents.run_store import (
    at_bound,
    fingerprint_sequences,
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

    def test_json_matches_json_dumps_and_rejects_nan(self, tmp_path):
        path = tmp_path / "out.json"
        write_json_atomically(path, {"b": 1, "a": [1.5, None]}, indent=2)
        assert path.read_text() == json.dumps({"b": 1, "a": [1.5, None]}, indent=2)
        with pytest.raises(ValueError):
            write_json_atomically(path, {"x": float("nan")}, indent=2)


class TestManifestDir:
    def test_named_by_hash_of_identity_and_holds_the_manifest(self, tmp_path):
        identity = {"b": 2, "a": [1, 2]}
        output = manifest_dir(tmp_path / "new" / "root", identity)
        text = json.dumps(identity, sort_keys=True, indent=2)
        assert output.name == hashlib.sha256(text.encode()).hexdigest()[:12]
        assert (output / "manifest.json").read_text() == text
        assert manifest_dir(tmp_path / "new" / "root", identity) == output

    def test_independent_of_key_order(self, tmp_path):
        first = manifest_dir(tmp_path, {"a": 1, "b": 2})
        assert manifest_dir(tmp_path, {"b": 2, "a": 1}) == first

    def test_provenance_is_written_but_not_hashed(self, tmp_path):
        identity = {"algorithm_version": 1}
        first = manifest_dir(tmp_path, identity, {"source": "text one"})
        second = manifest_dir(tmp_path, identity, {"source": "text two"})
        assert first == second
        manifest = json.loads((second / "manifest.json").read_text())
        assert manifest == {"algorithm_version": 1, "source": "text two"}

    def test_different_identities_get_different_directories(self, tmp_path):
        assert manifest_dir(tmp_path, {"v": 1}) != manifest_dir(tmp_path, {"v": 2})

    def test_provenance_may_not_shadow_identity(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate"):
            manifest_dir(tmp_path, {"v": 1}, {"v": 2})
        assert list(tmp_path.iterdir()) == []


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
