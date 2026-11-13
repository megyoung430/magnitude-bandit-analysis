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


PPD_STYLE = """Import a recording. The low_pass argument sets the
filter cutoff in Hz.  Returns a dictionary with the
following items:
    'filename'  - Data filename
    'mode'      - Acquisition mode
    For each analog signal:
        'analog_x'      - Raw signal (volts)
        'analog_x_filt' - Filtered signal
    For each digital signal:
        'digital_y' - Digital signal
"""


def test_a_returns_sentence_without_a_section_becomes_the_output_with_its_list():
    doc = parse_docstring(PPD_STYLE)
    assert doc.summary == "Import a recording."
    assert doc.body == "The low_pass argument sets the filter cutoff in Hz."
    assert doc.returns == (
        "Returns a dictionary with the following items:\n"
        "- 'filename' - Data filename\n"
        "- 'mode' - Acquisition mode\n"
        "- For each analog signal:\n"
        "  - 'analog_x' - Raw signal (volts)\n"
        "  - 'analog_x_filt' - Filtered signal\n"
        "- For each digital signal:\n"
        "  - 'digital_y' - Digital signal"
    )


def test_paragraphs_stay_separate_and_wrapped_lines_are_joined():
    doc = parse_docstring("Do a thing.\n\nFirst paragraph\nwraps here.\n\nSecond one.\n")
    assert doc.body == "First paragraph wraps here.\n\nSecond one."


def test_bullets_keep_their_lines_and_wrapped_bullets_are_joined():
    text = "Summary.\n\nSteps:\n\n- load the data\n  from disk\n- fit it\n1. numbered\n"
    assert parse_docstring(text).body == (
        "Steps:\n\n- load the data from disk\n- fit it\n- numbered"
    )


def test_a_returns_section_keeps_its_list():
    text = (
        "Summary.\n\nReturns:\n    DataFrame with columns:\n"
        "        mouse: The id.\n        p: The value.\n"
    )
    assert (
        parse_docstring(text).returns
        == "DataFrame with columns:\n- mouse: The id.\n- p: The value."
    )


def test_wrapped_lines_of_a_returns_section_are_joined():
    text = "Summary.\n\nReturns:\n    dict: The fit, with every\n        parameter.\n"
    assert parse_docstring(text).returns == "dict: The fit, with every parameter."


def test_a_short_first_paragraph_is_the_whole_summary():
    doc = parse_docstring("Fit the model. Uses several starts.\n\nMore.\n")
    assert doc.summary == "Fit the model. Uses several starts." and doc.body == "More."
