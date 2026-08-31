# Installation

## Requirements

AMLGraphX requires Python 3.12 or later. The project uses [uv](https://docs.astral.sh/uv/) for reproducible environments; it is the recommended workflow while the project is under active development.

## Install from the repository

Clone the repository and synchronize the default project environment:

```console
uv sync
```

Verify that the public package imports:

```console
uv run python -c "import amlgraphx; print(amlgraphx.__name__)"
```

When AMLGraphX releases are published to PyPI, users will be able to install a published wheel with `pip install amlgraphx`. Until then, installing from the repository is the supported path.

## Build the documentation locally

The documentation tools are an opt-in dependency group. Install them with:

```console
uv sync --group docs
```

Then render the site:

```console
uv run sphinx-build -W -b html docs docs/_build/html
```

`-W` treats warnings as build failures, which is appropriate for published documentation. Open `docs/_build/html/index.html` in a browser to inspect the result. For live editing, run:

```console
uv run sphinx-autobuild docs docs/_build/html
```

The generated `_build/` directory is not source documentation and is excluded from version control.
