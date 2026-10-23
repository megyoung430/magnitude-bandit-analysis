---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Cache-first runs, incremental cache refresh, cache check and pruning

Runs re-imported a whole cohort from the server whenever any session was added, because the
import cache was keyed on a scan of every raw `.tsv`. Now the cohort runs read the cache by
default, and the cache is brought up to date only when asked, parsing only the new or changed
files. A `check` command says whether the cache is up to date with the server and, if not, when
it was made and which subjects, sessions and files it lacks; `prune` removes old entries.

Results change: none (same trials); runs read the cache by default.

## Design

- Per file (`src/behavior_import/cache_store.py`): `parse_tsv_file` (read + JSON-expand one
  `.tsv`, which depends on that file alone) is cached under
  `import-files/<experiment>_rawdata_<cohort>/<key>.pkl`, the key a hash of the relative path,
  size, mtime_ns and `file_code_version()` (source of the parsing functions and the pandas
  version). The warnings a parse raises are stored and raised again on reuse, with the file's
  current path, so the import log is the same as a fresh import's.
- Per cohort: grouping, task-type classification, problem numbering and trial extraction need
  every session, so they run on the whole cohort at each refresh (`import_data(..., parse=...,
  paths=...)`), and the result is stored as a snapshot in `import-snapshots/<name>/` with a
  JSON `manifest.json` (root, classify flag, import and file code versions, created, file
  list). The snapshot is written to a `.tmp-` folder renamed into place last.
- `refresh_import_cache` (load_trials.py) loads the newest snapshot as is when its manifest
  matches the listing and the code; otherwise it assembles a new one and prunes the cohort.
  A failure to list or read the share raises `ShareReadError` (never a silent skip) and leaves
  the previous snapshot in use; files that are read but cannot be parsed are skipped as before
  and are not cached.
- Cache-first (`load_trials_with_log(..., from_cache=True)`): newest snapshot (exact root, else
  path tail), else the newest legacy joblib `_load` entry as before, else import and cache from
  the server. `load_trials_with_log(root)` refreshes.
- `cache_diff` (pure) and `cache_check.check_import_cache` / `describe_check`: the snapshot's
  manifest (or a legacy entry's recorded fingerprint) against the server listing, by subject
  and session. Exit code 1 also when only the import code changed.
- `cache_prune.prune_plan` (pure) and `apply_prune` (refuses paths outside the cache).

## Interfaces

- `scripts/full_pipelines/import_cache.py {check,refresh,prune}`.
- `run_cohorts.py` / `run_all_drifting_problems.py`: cache-first by default, `--refresh-cache`,
  `--no-cache` (all-drifting only), `--from-cache` accepted as a no-op.
- `import_data(..., parse=parse_tsv_file, paths=None)`, `ParsedFile`, `ImportAborted`.

## Acceptance criteria

- A refresh re-parses only new and changed files (counted) and the result equals a fresh
  import (trials, frames, source paths, import log with warnings).
- A cache-first run makes no os-level access to the raw folder and uses the newest snapshot;
  legacy and no-cache fallbacks work; `--refresh-cache` reads the server first.
- `check` groups new/changed/removed files by subject and session, exit 0/1/2.
- Prune keeps the newest snapshot and the only cache of a cohort, removes unreferenced per-file
  parses, and a dry run deletes nothing.
- An interrupted refresh or snapshot write leaves the old snapshot loading; nothing is written
  under the raw folder.
