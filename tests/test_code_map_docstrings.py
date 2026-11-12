"""Parsing Google-style docstrings into summary, body, arguments, returns and raises."""

from src.code_map.docstrings import parse_docstring

GOOGLE = """Fit the model to one mouse.

Runs several starts and keeps the best.

Args:
    trials (pd.DataFrame): One row per trial.
    n_starts: Number of optimizer starts,
        each from a random point.
    **kwargs: Passed to the optimizer.

Returns:
    FitResult: The best start.

Raises:
    ValueError: If *trials* is empty.
"""


def test_summary_and_body_are_the_first_and_remaining_paragraphs():
    doc = parse_docstring(GOOGLE)
    assert doc.summary == "Fit the model to one mouse."
    assert doc.body == "Runs several starts and keeps the best."


def test_arguments_keep_type_and_join_continuation_lines():
    doc = parse_docstring(GOOGLE)
    assert [a.name for a in doc.args] == ["trials", "n_starts", "kwargs"]
    assert doc.args[0].type == "pd.DataFrame"
    assert doc.args[1].type == ""
    assert doc.args[1].description == "Number of optimizer starts, each from a random point."


def test_returns_and_raises_are_flattened_to_one_line():
    doc = parse_docstring(GOOGLE)
    assert doc.returns == "FitResult: The best start."
    assert doc.raises == "ValueError: If *trials* is empty."


def test_a_missing_docstring_gives_empty_parts():
    doc = parse_docstring(None)
    assert (doc.summary, doc.body, doc.args, doc.returns, doc.raises) == ("", "", (), "", "")


def test_a_one_line_docstring_is_only_a_summary():
    doc = parse_docstring("Mean ignoring NaNs.")
    assert doc.summary == "Mean ignoring NaNs." and doc.body == ""


def test_yields_counts_as_the_output():
    doc = parse_docstring("Walk the sessions.\n\nYields:\n    dict: One session.\n")
    assert doc.returns == "dict: One session."
