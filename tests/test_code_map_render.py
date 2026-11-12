"""Rendering the page and reading the repo for the code map."""

import json
import subprocess

import pytest

from src.code_map.files import git_meta, read_records, source_paths
from src.code_map.render import PLACEHOLDER, render_html, template_text


def test_the_data_is_embedded_as_json_and_cannot_close_the_script_tag():
    html = render_html({"intro": "a </script> b"}, f"<script>{PLACEHOLDER}</script>")
    inner = html.removeprefix("<script>").removesuffix("</script>")
    assert "</script>" not in inner
    assert json.loads(inner.replace("<\\/", "</")) == {"intro": "a </script> b"}


def test_a_template_without_the_placeholder_fails_loudly():
    with pytest.raises(ValueError, match=PLACEHOLDER):
        render_html({}, "<p>no slot</p>")


def test_the_shipped_template_has_a_title_and_the_placeholder_once():
    text = template_text()
    assert text.count(PLACEHOLDER) == 1 and text.startswith("<title>")


def make_repo(root):
    (root / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "src" / "pkg" / "a.py").write_text('"""A."""\n\n\ndef f():\n    """Do f."""\n')
    (root / "src" / "pkg" / "__pycache__" / "a.py").write_text("junk(")
    (root / "scripts" / ".ipynb_checkpoints").mkdir(parents=True)
    (root / "scripts" / "run.py").write_text("print(1)\n")
    notebook = {"cells": [{"cell_type": "markdown", "source": ["# Hi"]}]}
    (root / "scripts" / "nb.ipynb").write_text(json.dumps(notebook))
    (root / "scripts" / ".ipynb_checkpoints" / "nb-checkpoint.ipynb").write_text("{}")
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("def test(): pass\n")


def test_sources_are_src_and_scripts_without_caches_checkpoints_or_tests(tmp_path):
    make_repo(tmp_path)
    python, notebooks = source_paths(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in python] == [
        "scripts/run.py",
        "src/pkg/a.py",
    ]
    assert [p.relative_to(tmp_path).as_posix() for p in notebooks] == ["scripts/nb.ipynb"]


def test_read_records_parses_every_source_with_repo_relative_paths(tmp_path):
    make_repo(tmp_path)
    modules, notebooks = read_records(tmp_path)
    assert [m.path for m in modules] == ["scripts/run.py", "src/pkg/a.py"]
    assert modules[1].defs[0].doc.summary == "Do f."
    assert notebooks[0].blocks[0].heading == "Hi"


def test_git_meta_names_branch_commit_and_date(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "trunk")
    (tmp_path / "f").write_text("x")
    git("add", "f")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    meta = git_meta(tmp_path)
    assert meta.startswith("trunk @ ") and len(meta.split(", ")) == 2


def test_git_meta_outside_a_repository_fails_loudly(tmp_path):
    with pytest.raises(RuntimeError, match="git"):
        git_meta(tmp_path)
