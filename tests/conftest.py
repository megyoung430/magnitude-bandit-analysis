"""Fixtures shared across test files."""

from pathlib import Path

import pytest


@pytest.fixture
def edit_module_text(monkeypatch):
    """Return edit(module, old, new): Path.read_text then yields *module*'s source edited.

    Manifests record module source text as provenance; this simulates editing a
    module without touching the file.
    """

    def edit(module, old, new):
        module_path = Path(module.__file__)
        read_text = Path.read_text
        assert old in read_text(module_path)

        def patched(self, *args, **kwargs):
            text = read_text(self, *args, **kwargs)
            return text.replace(old, new) if self == module_path else text

        monkeypatch.setattr(Path, "read_text", patched)

    return edit
