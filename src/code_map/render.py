"""The code-map page: the template with the data embedded as JSON (pure apart from reading the
template that ships with this package)."""

import json
from importlib import resources

PLACEHOLDER = "__DATA__"


def template_text() -> str:
    """The page template shipped next to this module."""
    return resources.files("src.code_map").joinpath("template.html").read_text(encoding="utf-8")


def render_html(data: dict, template: str) -> str:
    """*template* with *data* embedded as JSON in place of the placeholder.

    ``</`` is written as ``<\\/`` (still valid JSON) so a docstring that contains ``</script>``
    cannot end the script element early.

    Raises:
        ValueError: If the template does not contain the placeholder exactly once.
    """
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"the template must contain {PLACEHOLDER} exactly once")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)
