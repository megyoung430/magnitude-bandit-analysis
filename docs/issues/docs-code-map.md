# docs-code-map: a searchable map of the repo

## What
`scripts/build_code_map.py` writes `results/code_map/magnitude-bandit-code-map.html`: one self-contained
page that lists every folder, file and notebook block, and every function, class and method with its
signature, inputs (name, type, description), output, errors raised and what it does. A search box
filters across all of them.

## How
- `src/code_map/extract.py` reads each file with `ast` (signatures, nesting, constants) and each
  notebook's cells (blocks under its markdown headings, with the functions each block calls).
- `src/code_map/docstrings.py` splits Google-style docstrings into summary, body, Args, Returns and
  Raises.
- `src/code_map/assemble.py` joins them: an input's type is its annotation, else the docstring's type;
  the output is the Returns section, else the return annotation.
- `src/code_map/render.py` embeds the data as JSON in `src/code_map/template.html`; `files.py` is the
  I/O edge (reading the repo, the notes and the git commit; writing the page).

## Hand-written inputs (`docs/code_map/`)
- `descriptions.tsv`: `path::qualname<TAB>text` for definitions without a docstring. The page marks
  these "from code". A docstring takes precedence once written, so the preferred fix for an
  undescribed function is a docstring.
- `overviews.toml`: the page intro, the repo-layout section, folder summaries (their order is the
  page's folder order), and summaries for notebooks and for files whose docstring is missing.

The script prints definitions with neither a docstring nor a description, and TSV lines that match no
definition in the checkout (renamed, removed, or on a branch that is not checked out); `--strict` exits
with an error when anything is undescribed. Nested functions need no description.

## Published page
`.github/workflows/code-map.yml` rebuilds the page on every push to main and deploys it with GitHub
Pages (<https://megyoung430.github.io/magnitude-bandit-analysis/>); the Methods & Docs page in the lab
Notion embeds it, so the embedded map always matches main. The build needs only the standard library
and pydantic-settings, so it skips installing the analysis environment. Pages must be enabled once in
the repository settings (Settings > Pages > Source: GitHub Actions).

## Notion
`--notion` also syncs the "magnitude-bandit-analysis Code Map" database (Value, Schema, MRN >
Methods & Docs > Behavior > Data Analysis; its id is in `docs/code_map/notion.toml`). One row per
file (folder, type, summary, lines, definitions); each row's page has the summary and one toggle
per function or class with its description, inputs, output, errors and nested definitions.

- `src/code_map/notion_blocks.py` builds the API JSON (rich text chunked at 2000 characters,
  toggles with flat children, at most 100 blocks / 900 nested blocks per request).
- `src/code_map/notion_sync.py` compares each row's "Source Hash" with the file's content and
  creates, replaces (new row written and fingerprinted first, then the old one trashed) or trashes
  rows; unchanged files cost nothing. A changed file gets a new row, so links to its old row break.
- `src/code_map/notion_api.py` is a standard-library client with retries on rate limits.

Setup: create an internal integration at notion.so/profile/integrations, add it to the Methods &
Docs page (••• > Connections), and set `NOTION_TOKEN` in `.env`. `--notion --dry-run` lists the
changes without writing. Edits made in Notion are overwritten by the next sync: change the
docstrings or `docs/code_map/` instead.

## Results change
None: documentation only.
