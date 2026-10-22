"""The notes that mark outputs made from the import cache without the server."""

from datetime import datetime

from src.behavior_import.import_log import CacheSource, ImportLog
from src.pipelines.cache_provenance import (
    FROM_CACHE_FILE,
    cache_sources,
    from_cache_description,
    write_from_cache_notes,
)

OCT_1 = datetime(2026, 10, 1, 9, 30).timestamp()
OCT_5 = datetime(2026, 10, 5, 17, 0).timestamp()


def source(time=OCT_1, **kwargs):
    return CacheSource("/share/exp/rawdata/cohort-03", time, True, 120, **kwargs)


def test_only_logs_loaded_from_the_cache_give_a_source():
    cached = ImportLog(cache_source=source())
    sources = cache_sources({"cohort-01": ImportLog(), "cohort-03": cached, "cohort-04": None})
    assert sources == {"cohort-03": source()}


def test_the_description_lists_each_cohort():
    text = from_cache_description(
        {
            "cohort-01": source(OCT_5)._replace(code_matches=False, root="/r/cohort-01"),
            "cohort-03": source(matched_on_tail=True, filled_fields=("port_finding_sessions",)),
        }
    )
    assert "sessions added since are missing" in text
    (one,) = [line for line in text.splitlines() if line.startswith("cohort-01")]
    assert "2026-10-05 17:00" in one and "older import code" in one and "120 .tsv files" in one
    (three,) = [line for line in text.splitlines() if line.startswith("cohort-03")]
    assert "current import code" in three and "last 3 path parts" in three
    assert "port_finding_sessions" in three


def test_the_note_is_written_into_each_folder_and_removed_by_a_run_from_the_server(tmp_path):
    folders = [tmp_path / "figures", tmp_path / "tables" / "deep"]
    write_from_cache_notes(folders, {"cohort-03": source()})
    for folder in folders:
        assert "cohort-03" in (folder / FROM_CACHE_FILE).read_text()
    write_from_cache_notes(folders, {})
    assert not any((folder / FROM_CACHE_FILE).exists() for folder in folders)


def test_a_run_from_the_server_creates_no_folders(tmp_path):
    write_from_cache_notes([tmp_path / "never-written"], {})
    assert not (tmp_path / "never-written").exists()
