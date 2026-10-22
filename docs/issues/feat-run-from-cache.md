---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Run the pooled cohort pipelines from the import cache without the share

`run_cohorts.py --cohorts 3 --pooled-only --paper-ready` and
`run_all_drifting_problems.py --paper-ready` need the share today: the import cache is keyed on
a scan of every raw `.tsv` and on the hash of `src/behavior_import`, so an unmounted share (or
any import-code change since the cache was filled) means a full re-import. `--from-cache` loads
the newest cached import of each cohort instead, without touching the share.

Results change: none unless --from-cache is passed.

## Interfaces

- `load_trials_with_log(..., from_cache=False)`; `choose_cache_entry(entries, root,
  classify_task_types, code_version)` (pure); `cached_import_entries(cache_dir)` (I/O);
  `StaleCacheWarning`, `CachedImportError` in `src/behavior_import/load_trials.py`.
- `UncheckedSettings` / `get_unchecked_settings()` in `src/settings.py`.
- `from_cache` on `load_cohort`, `run_cohort`, `run_cohort_incremental`, the two `plan_*`
  functions, `check_exclusion_in_cohorts`, `run_cohorts` (via its settings) and
  `run_all_cohorts_drifting`.
- `--from-cache` on both scripts; exclusive with `--no-cache` in `run_all_drifting_problems.py`.
- `ImportLog.cache_source` (`CacheSource`); `src/pipelines/cache_provenance.py` writes
  `from-cache.txt` and the caption line; `plot_helpers.caption_note`.

## Acceptance criteria

- Selection prefers the current import code, else the newest; ignores other roots and the other
  classify flag; no match raises `FileNotFoundError` listing what is cached.
- A from-cache load calls no `is_dir`/`exists`/`stat`/`rglob`/`resolve`/`iterdir`/`glob` on the
  root, and works with the raw folder gone.
- The `StaleCacheWarning` gives the date, the number of `.tsv` files, whether the import code
  matches, and that newer sessions are missing; it is in the import log and `warnings_log.csv`.
- A cached log missing a field with a default gets the default (named in the warning); one
  without a default, or an entry that cannot be unpickled, raises `CachedImportError` naming it;
  no fall-through to the share. Unreadable metadata skips that entry with a warning.
- With no exact root match, a unique match on the last three path parts is used and the warning
  says so; several such roots raise and list them.
- No os-level access (`stat`, `lstat`, `listdir`, `scandir`, `access`, `open`) under the data
  root during the from-cache cohort, dry-run and all-cohorts runs.
- `from-cache.txt` in every output folder of a from-cache run (cohort figures and tables,
  `paper-ready/<cohort>/`, `all-cohorts[-without-…]/` figures and tables, and their paper-ready
  folders), and the caption line in every paper caption; neither in a run from the server.
- The cohort pooled paper-ready run and the all-cohorts paper-ready run give identical CSVs and
  PNGs (bar `warnings_log.csv`) from the cache with the raw folder renamed away, also after an
  import-code change.

## Decisions

- The root is compared with `os.path.abspath`, never `resolve()`: resolving queries every folder
  on the path, which stalls on a hung mount. A cache filled through another mount point or a
  symlinked path is found by the tail fallback.
- `--exclude-mice` with `--from-cache` reuses the check's load for the run; from the server the
  cohort is still reloaded so that the run sees current files and memory holds one cohort.
- The warning is shown on every from-cache load (filter `"always"`), so each run's log has it.
- Metadata of incomplete joblib entries (no `output.pkl` or `metadata.json`) is skipped;
  unparseable metadata of a complete entry raises.
- Changing `load_trials.py` changes `import_code_version()`, so the first normal run after this
  merges re-imports from the share once (as any import-code change does).
