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

## Results change
None: documentation only.
