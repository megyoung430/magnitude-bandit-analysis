"""``cache_diff``: what the server has that a cache snapshot lacks, by subject and session."""

from src.behavior_import.cache_diff import cache_diff, describe_diff, grouped_changes
from src.behavior_import.cache_store import ServerFile

S1 = "sub-01_id-M1/ses-1_date-20260101/behav"
S2 = "sub-01_id-M1/ses-2_date-20260102/behav"


def f(path, size=10, mtime=1):
    return ServerFile(path, size, mtime)


CACHED = (f(f"{S1}/a.tsv"), f(f"{S2}/b.tsv"))


def statuses(diff):
    return [(c.status, c.path) for c in diff.changes]


def test_identical_listings_are_up_to_date():
    diff = cache_diff(CACHED, tuple(reversed(CACHED)))
    assert diff.up_to_date and diff.changes == ()
    assert describe_diff(diff) == []


def test_a_new_subject_is_new_with_its_sessions():
    new = f("sub-02_id-M2/ses-1_date-20260103/behav/c.tsv")
    diff = cache_diff(CACHED, (*CACHED, new))
    assert statuses(diff) == [("new", new.path)]
    assert diff.new_subjects == ("M2",)
    assert diff.new_sessions == (("M2", "ses-1_date-20260103"),)
    assert not diff.up_to_date


def test_a_new_session_of_a_known_subject():
    new = f("sub-01_id-M1/ses-3_date-20260104/behav/c.tsv")
    diff = cache_diff(CACHED, (*CACHED, new))
    assert diff.new_subjects == ()
    assert diff.new_sessions == (("M1", "ses-3_date-20260104"),)


def test_a_new_file_in_a_cached_session_is_new_but_the_session_is_not():
    new = f(f"{S1}/a2.tsv")
    diff = cache_diff(CACHED, (*CACHED, new))
    assert statuses(diff) == [("new", new.path)]
    assert diff.new_sessions == () and diff.new_subjects == ()


def test_a_changed_size_or_mtime_is_changed_with_both_states():
    server = (f(f"{S1}/a.tsv", mtime=2), f(f"{S2}/b.tsv", size=11))
    diff = cache_diff(CACHED, server)
    assert statuses(diff) == [("changed", f"{S1}/a.tsv"), ("changed", f"{S2}/b.tsv")]
    first = diff.changes[0]
    assert (first.cached, first.server) == (CACHED[0], server[0])


def test_a_file_gone_from_the_server_is_removed():
    diff = cache_diff(CACHED, CACHED[:1])
    assert statuses(diff) == [("removed", f"{S2}/b.tsv")]
    assert diff.changes[0].server is None


def test_the_changes_are_grouped_by_subject_then_session():
    server = (
        f(f"{S1}/a.tsv", mtime=2),
        f("sub-02_id-M2/ses-1_date-20260103/behav/c.tsv"),
        f("loose.tsv"),
    )
    grouped = grouped_changes(cache_diff(CACHED, server))
    assert list(grouped) == ["(no subject folder)", "M1", "M2"]
    assert list(grouped["M1"]) == ["ses-1_date-20260101", "ses-2_date-20260102"]
    assert [c.status for c in grouped["M1"]["ses-2_date-20260102"]] == ["removed"]
    assert list(grouped["(no subject folder)"]) == ["(no session folder)"]


def test_the_description_marks_new_subjects_and_sessions_and_names_each_file():
    server = (
        f(f"{S1}/a.tsv", size=20, mtime=2),
        f("sub-02_id-M2/ses-1_date-20260103/behav/c.tsv"),
        f("sub-01_id-M1/ses-3_date-20260104/behav/d.tsv"),
    )
    lines = describe_diff(cache_diff(CACHED, server))
    text = "\n".join(lines)
    assert "M2 (new subject)" in text
    assert "ses-3_date-20260104 (new session)" in text
    assert "new      behav/c.tsv" in text
    assert "changed  behav/a.tsv (size 10 -> 20 bytes" in text
    assert "removed  behav/b.tsv" in text
    # subject, then its sessions, then their files, each indented one step further
    m1 = lines.index("M1")
    assert lines[m1 + 1].strip() == "ses-1_date-20260101"
    assert lines[m1 + 2].startswith("    changed")
