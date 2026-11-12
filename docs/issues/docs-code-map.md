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

## In plain words
`docs/code_map/explanations.md` holds plain-language text for readers new to the methods, keyed by
file path (a primer: what question the file answers, the method from scratch, how to read its
results) or `path::name` (one function). It covers every file of `behavior_analysis`, `agents`,
`funcimg_analysis` and `photometry_analysis`, and the functions that the pipelines, scripts and
notebooks call, except for small utilities whose one-line description says enough. The page shows
each entry as a dropdown (open for a file's primer, closed for a function), and search includes it.
The build lists entries that match no file or function; update an entry when the code it describes
changes.

## Published page
`.github/workflows/code-map.yml` rebuilds the page on every push to main and deploys it with GitHub
Pages (<https://megyoung430.github.io/magnitude-bandit-analysis/>); the Methods & Docs page in the lab
Notion embeds it, so the embedded map always matches main. The build needs only the standard library
and pydantic-settings, so it skips installing the analysis environment. Pages must be enabled once in
the repository settings (Settings > Pages > Source: GitHub Actions).

## Results change
None: documentation only.
