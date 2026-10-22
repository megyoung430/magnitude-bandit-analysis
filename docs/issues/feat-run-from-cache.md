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
- `--from-cache` on both scripts; exclusive with `--no-cache`.

## Acceptance criteria

- Selection prefers the current import code, else the newest; ignores other roots and the other
  classify flag; no match raises `FileNotFoundError` listing what is cached.
- A from-cache load calls no `is_dir`/`exists`/`stat`/`rglob`/`resolve`/`iterdir`/`glob` on the
  root, and works with the raw folder gone.
- The `StaleCacheWarning` gives the date, the number of `.tsv` files, whether the import code
  matches, and that newer sessions are missing; it is in the import log and `warnings_log.csv`.
- An unreadable or outdated cache entry raises `CachedImportError` naming it; no fall-through to
  the share.
- The cohort pooled paper-ready run and the all-cohorts paper-ready run give identical CSVs and
  PNGs (bar `warnings_log.csv`) from the cache with the raw folder renamed away, also after an
  import-code change.

## Decisions

- The root is compared with `os.path.abspath`, never `resolve()`: resolving queries every folder
  on the path, which stalls on a hung mount. A cache filled through a symlinked path is therefore
  found only via that same resolved path.
- The warning is shown on every from-cache load (filter `"always"`), so each run's log has it.
- Metadata of incomplete joblib entries (no `output.pkl` or `metadata.json`) is skipped;
  unparseable metadata of a complete entry raises.
- Changing `load_trials.py` changes `import_code_version()`, so the first normal run after this
  merges re-imports from the share once (as any import-code change does).
