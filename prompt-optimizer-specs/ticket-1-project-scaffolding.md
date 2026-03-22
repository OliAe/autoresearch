# Ticket 1: Project Scaffolding & Configuration

## Overview

Set up the Python project "Prompt Optimizer" with proper packaging, configuration, and foundational types. This is an eval-driven LLM program optimization system. This ticket creates the entire project skeleton — directory structure, packaging via `pyproject.toml` with `uv`, global configuration, CLI entry point, and placeholder modules — so that all subsequent tickets can build on a clean, working foundation.

After this ticket is complete, a developer (or AI) should be able to:
- Install the project in editable mode with `uv sync`
- Run `uv run prompt-opt --help` and see all subcommands listed
- Run `uv run pytest` with a passing test suite
- Run `uv run ruff check .` with zero errors
- Import `from prompt_optimizer.config import Settings` and get a working Pydantic settings object

---

## Acceptance Criteria

- [ ] Project uses `pyproject.toml` with `uv` as the build tool (hatchling backend)
- [ ] Python 3.11+ required (`requires-python = ">=3.11"`)
- [ ] Package name: `prompt-optimizer`, importable as `prompt_optimizer`
- [ ] Runtime dependencies: `anthropic`, `pydantic>=2.0`, `pydantic-settings>=2.0`, `click`, `pyyaml`, `rich`
- [ ] Dev dependencies (in `[dependency-groups]`): `pytest`, `pytest-asyncio`, `ruff`
- [ ] CLI entry point: `prompt-opt` command registered via `[project.scripts]` in pyproject.toml
- [ ] `src/` layout used — all package code lives under `src/prompt_optimizer/`
- [ ] All `__init__.py` files created for every package/subpackage
- [ ] Global config system using Pydantic `BaseSettings` (from `pydantic-settings`)
- [ ] `.env` file support for API keys (`ANTHROPIC_API_KEY`)
- [ ] `.env.example` template file provided
- [ ] All placeholder modules created with descriptive docstrings explaining their future purpose
- [ ] Basic CLI skeleton with Click group and placeholder subcommands (10 subcommands total)
- [ ] `uv run pytest` runs successfully and passes
- [ ] `uv run ruff check .` passes with no errors
- [ ] `.gitignore` includes all relevant patterns
- [ ] `tests/conftest.py` has basic reusable fixtures

---

## Directory Structure

After this ticket is complete, the project tree must look exactly like this:

```
prompt-optimizer/
├── pyproject.toml
├── .env.example
├── .gitignore
├── src/
│   └── prompt_optimizer/
│       ├── __init__.py
│       ├── config.py
│       ├── cli.py
│       ├── dataset.py
│       ├── candidate.py
│       ├── runner.py
│       ├── judge.py
│       ├── optimizer.py
│       ├── controller.py
│       ├── history.py
│       └── providers/
│           ├── __init__.py
│           └── anthropic.py
└── tests/
    ├── __init__.py
    └── conftest.py
```

---

## Detailed Design — File-by-File Contents

Below is the **exact content** for every file. Create each file verbatim.

---

### 1. `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "prompt-optimizer"
version = "0.1.0"
description = "An eval-driven LLM program optimization system"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "anthropic>=0.39.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "click>=8.0",
    "pyyaml>=6.0",
    "rich>=13.0",
]

[project.scripts]
prompt-opt = "prompt_optimizer.cli:cli"

[tool.hatch.build.targets.wheel]
packages = ["src/prompt_optimizer"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
filterwarnings = [
    "ignore::DeprecationWarning",
]

[tool.ruff]
target-version = "py311"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.isort]
known-first-party = ["prompt_optimizer"]
```

---

### 2. `src/prompt_optimizer/__init__.py`

```python
"""Prompt Optimizer — an eval-driven LLM program optimization system.

This package provides tools for systematically optimizing LLM prompts
through evaluation-driven experimentation. It manages datasets, generates
candidate prompt variations, runs them against evaluation criteria, and
tracks results over time.
"""

__version__ = "0.1.0"
```

---

### 3. `src/prompt_optimizer/config.py`

```python
"""Global configuration for Prompt Optimizer.

This module provides the Settings class which centralizes all configuration
for the application. Settings are loaded from environment variables and/or
a .env file. The PROMPT_OPT_ prefix is used for all environment variables
(e.g., PROMPT_OPT_DEFAULT_MODEL).

The one exception is ANTHROPIC_API_KEY, which is read directly without
the prefix for convenience and compatibility with the Anthropic SDK.

Usage:
    from prompt_optimizer.config import get_settings

    settings = get_settings()
    print(settings.default_model)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    All settings can be overridden via environment variables prefixed with
    PROMPT_OPT_ (e.g., PROMPT_OPT_DEFAULT_MODEL=claude-sonnet-4-20250514).

    The anthropic_api_key field is intentionally allowed to be empty at
    construction time. It is only validated when an API call is actually
    needed, so that CLI commands that don't require the API (like --help,
    config, etc.) still work without a key set.
    """

    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key. Can also be set as ANTHROPIC_API_KEY without prefix.",
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default Anthropic model to use for prompt evaluation.",
    )
    default_temperature: float = Field(
        default=0.0,
        description="Default temperature for LLM calls.",
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent API requests.",
    )
    datasets_dir: Path = Field(
        default=Path("datasets"),
        description="Directory where datasets are stored.",
    )
    runs_dir: Path = Field(
        default=Path("runs"),
        description="Directory where run results are stored.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PROMPT_OPT_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def require_api_key(self) -> str:
        """Return the API key, raising an error if it is not set.

        Call this method before making any API requests. It ensures the key
        is present and returns it.

        Returns:
            The Anthropic API key string.

        Raises:
            ValueError: If no API key has been configured.
        """
        if not self.anthropic_api_key:
            raise ValueError(
                "Anthropic API key is required but not set. "
                "Set it via PROMPT_OPT_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY "
                "in your environment or .env file."
            )
        return self.anthropic_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached global settings instance.

    This function caches the Settings object so it is only constructed once.
    All callers receive the same instance.

    Returns:
        The global Settings instance.
    """
    return Settings()
```

---

### 4. `src/prompt_optimizer/cli.py`

```python
"""Command-line interface for Prompt Optimizer.

This module defines the Click CLI group and all subcommands. Each subcommand
is a placeholder that will be implemented in subsequent tickets.

The CLI is registered as the `prompt-opt` console script entry point.

Usage:
    prompt-opt --help
    prompt-opt init my-project
    prompt-opt run
"""

from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="prompt-optimizer")
def cli() -> None:
    """Prompt Optimizer — eval-driven LLM program optimization."""


@cli.command()
@click.argument("name")
def init(name: str) -> None:
    """Initialize a new prompt optimization project.

    Creates the project directory structure, default configuration file,
    and an empty dataset ready for examples to be added.

    NAME is the name of the project to create.
    """
    console.print(f"[bold green]Initializing project:[/bold green] {name}")
    console.print("[dim]Not yet implemented.[/dim]")


@cli.command()
@click.argument("dataset")
@click.option("--input", "-i", "input_text", required=True, help="Input text for the example.")
@click.option("--expected", "-e", required=True, help="Expected output for the example.")
def add(dataset: str, input_text: str, expected: str) -> None:
    """Add a single example to a dataset.

    DATASET is the name of the target dataset.
    """
    console.print(f"[bold green]Adding example to dataset:[/bold green] {dataset}")
    console.print("[dim]Not yet implemented.[/dim]")


@cli.command(name="import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", default="jsonl", help="File format (jsonl, csv, yaml).")
def import_data(file: str, fmt: str) -> None:
    """Import examples from a file into a dataset.

    FILE is the path to the data file to import.
    """
    console.print(f"[bold green]Importing from:[/bold green] {file} (format: {fmt})")
    console.print("[dim]Not yet implemented.[/dim]")


@cli.command()
@click.argument("dataset")
@click.option("--train-ratio", default=0.8, help="Fraction of data to use for training.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def split(dataset: str, train_ratio: float, seed: int) -> None:
    """Split a dataset into train/test sets.

    DATASET is the name of the dataset to split.
    """
    console.print(f"[bold green]Splitting dataset:[/bold green] {dataset}")
    console.print("[dim]Not yet implemented.[/dim]")


@cli.command()
@click.option("--show", is_flag=True, help="Display current configuration.")
def config(show: bool) -> None:
    """View or update configuration settings."""
    if show:
        console.print("[bold]Current configuration:[/bold]")
        console.print("[dim]Not yet implemented.[/dim]")
    else:
        console.print("[dim]Use --show to display current config.[/dim]")


@cli.command()
@click.option("--dataset", "-d", required=True, help="Dataset to evaluate against.")
@click.option("--model", "-m", default=None, help="Model to use (overrides config).")
@click.option("--concurrency", "-c", default=None, type=int, help="Max concurrent requests.")
def run(dataset: str, model: str | None, concurrency: int | None) -> None:
    """Run prompt optimization against a dataset.

    Executes the current prompt candidates against the dataset, evaluates
    results, and records the run.
    """
    console.print(f"[bold green]Starting run on dataset:[/bold green] {dataset}")
    console.print("[dim]Not yet implemented.[/dim]")


@cli.command()
def status() -> None:
    """Show the status of the current optimization project.

    Displays information about datasets, candidates, and recent runs.
    """
    console.print("[bold]Project status:[/bold]")
    console.print("[dim]Not yet implemented.[/dim]")


@cli.command()
@click.option("--run-id", "-r", default=None, help="Specific run ID to show results for.")
@click.option("--latest", is_flag=True, help="Show results for the most recent run.")
def results(run_id: str | None, latest: bool) -> None:
    """Display results from optimization runs.

    Shows metrics, scores, and comparisons between prompt candidates.
    """
    console.print("[bold]Results:[/bold]")
    console.print("[dim]Not yet implemented.[/dim]")


@cli.command()
@click.option("--run-id", "-r", default=None, help="Specific run ID to generate report for.")
@click.option("--output", "-o", default=None, help="Output file path for the report.")
def report(run_id: str | None, output: str | None) -> None:
    """Generate a detailed report from optimization runs.

    Produces a comprehensive analysis including metrics, comparisons,
    and recommendations.
    """
    console.print("[bold]Generating report...[/bold]")
    console.print("[dim]Not yet implemented.[/dim]")


@cli.command()
@click.option("--run-id", "-r", default=None, help="Specific run ID to export.")
@click.option("--format", "-f", "fmt", default="json", help="Export format (json, csv, yaml).")
@click.option("--output", "-o", required=True, help="Output file path.")
def export(run_id: str | None, fmt: str, output: str) -> None:
    """Export results or data to a file.

    Exports run results, datasets, or prompt candidates in the specified format.
    """
    console.print(f"[bold green]Exporting to:[/bold green] {output} (format: {fmt})")
    console.print("[dim]Not yet implemented.[/dim]")
```

---

### 5. `src/prompt_optimizer/dataset.py`

```python
"""Dataset management for Prompt Optimizer.

This module will handle loading, saving, and manipulating datasets of
input/output examples used for prompt evaluation. Key responsibilities:

- Defining the Dataset and Example data models (Pydantic)
- Loading datasets from JSONL, CSV, and YAML files
- Splitting datasets into train/test sets
- Validating dataset structure and contents
- Persisting datasets to disk in a standard format

Implemented in Ticket 2.
"""
```

---

### 6. `src/prompt_optimizer/candidate.py`

```python
"""Prompt candidate management for Prompt Optimizer.

This module will define and manage prompt candidates — the different prompt
variations being evaluated. Key responsibilities:

- Defining the Candidate data model (system prompt text, metadata, lineage)
- Generating new candidate variations from existing prompts
- Tracking candidate history and parent-child relationships
- Serializing/deserializing candidates to/from disk

Implemented in Ticket 3.
"""
```

---

### 7. `src/prompt_optimizer/runner.py`

```python
"""Execution runner for Prompt Optimizer.

This module will handle running prompt candidates against datasets by
making LLM API calls. Key responsibilities:

- Executing a candidate prompt against all examples in a dataset
- Managing concurrent API requests with rate limiting
- Collecting raw responses and timing information
- Handling retries and error recovery
- Reporting progress during execution

Implemented in Ticket 4.
"""
```

---

### 8. `src/prompt_optimizer/judge.py`

```python
"""Evaluation and judging for Prompt Optimizer.

This module will score and evaluate the outputs produced by running
candidates against datasets. Key responsibilities:

- Defining scoring criteria and rubrics
- LLM-as-judge evaluation (using a separate LLM call to score outputs)
- Exact-match and fuzzy-match scoring
- Custom scoring function support
- Aggregating scores across examples into summary metrics

Implemented in Ticket 5.
"""
```

---

### 9. `src/prompt_optimizer/optimizer.py`

```python
"""Optimization strategy for Prompt Optimizer.

This module will implement the core optimization loop that generates
improved prompt candidates based on evaluation results. Key responsibilities:

- Analyzing evaluation results to identify weaknesses
- Generating improved prompt candidates using an LLM
- Implementing different optimization strategies (hill-climbing, tournament, etc.)
- Managing the optimization budget (max iterations, API cost limits)

Implemented in Ticket 6.
"""
```

---

### 10. `src/prompt_optimizer/controller.py`

```python
"""Orchestration controller for Prompt Optimizer.

This module will coordinate the end-to-end optimization workflow,
connecting all other components. Key responsibilities:

- Orchestrating the run-evaluate-optimize loop
- Managing state across iterations
- Handling stop conditions (convergence, budget exhaustion)
- Coordinating between runner, judge, and optimizer modules
- Emitting events/callbacks for progress reporting

Implemented in Ticket 7.
"""
```

---

### 11. `src/prompt_optimizer/history.py`

```python
"""Run history and persistence for Prompt Optimizer.

This module will manage the storage and retrieval of optimization run
history. Key responsibilities:

- Saving run results (candidates, scores, metadata) to disk
- Loading and querying past runs
- Comparing results across runs
- Generating run IDs and managing the runs directory structure
- Exporting history in various formats (JSON, CSV)

Implemented in Ticket 8.
"""
```

---

### 12. `src/prompt_optimizer/providers/__init__.py`

```python
"""LLM provider integrations for Prompt Optimizer.

This subpackage contains provider-specific client wrappers for making
LLM API calls. Each provider module adapts a specific API client to
a common interface used by the runner module.
"""
```

---

### 13. `src/prompt_optimizer/providers/anthropic.py`

```python
"""Anthropic API provider for Prompt Optimizer.

This module will wrap the Anthropic Python SDK to provide a consistent
interface for making Claude API calls. Key responsibilities:

- Initializing the Anthropic client with API key from settings
- Sending prompt completion requests (sync and async)
- Handling rate limiting, retries, and error mapping
- Streaming support for long-running completions
- Token usage tracking

Implemented in Ticket 4 (alongside runner.py).
"""
```

---

### 14. `tests/__init__.py`

```python
```

(Empty file — no content, just needs to exist.)

---

### 15. `tests/conftest.py`

```python
"""Shared pytest fixtures for Prompt Optimizer tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from prompt_optimizer.config import Settings


@pytest.fixture()
def tmp_datasets_dir(tmp_path: Path) -> Path:
    """Create and return a temporary datasets directory.

    Use this fixture when tests need to read/write dataset files
    without affecting the real filesystem.

    Returns:
        A Path to a temporary directory for datasets.
    """
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    return datasets_dir


@pytest.fixture()
def tmp_runs_dir(tmp_path: Path) -> Path:
    """Create and return a temporary runs directory.

    Use this fixture when tests need to read/write run result files
    without affecting the real filesystem.

    Returns:
        A Path to a temporary directory for runs.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return runs_dir


@pytest.fixture()
def mock_settings(tmp_datasets_dir: Path, tmp_runs_dir: Path) -> Settings:
    """Create a Settings instance with test-safe defaults.

    This fixture provides a Settings object that uses temporary directories
    and a fake API key, ensuring tests never touch real configuration or
    make real API calls.

    Returns:
        A Settings instance configured for testing.
    """
    return Settings(
        anthropic_api_key="test-api-key-not-real",
        default_model="claude-sonnet-4-20250514",
        default_temperature=0.0,
        max_concurrent_requests=2,
        datasets_dir=tmp_datasets_dir,
        runs_dir=tmp_runs_dir,
    )


@pytest.fixture()
def _patch_settings(mock_settings: Settings):
    """Patch get_settings() to return mock_settings.

    Use this fixture (via @pytest.mark.usefixtures("_patch_settings"))
    when code under test calls get_settings() and you want it to
    receive the test-safe settings.
    """
    with patch("prompt_optimizer.config.get_settings", return_value=mock_settings):
        yield
```

---

### 16. `.env.example`

```
# Prompt Optimizer configuration
# Copy this file to .env and fill in your values.

# Required: Your Anthropic API key
PROMPT_OPT_ANTHROPIC_API_KEY=your-api-key-here

# Optional: Override default model
# PROMPT_OPT_DEFAULT_MODEL=claude-sonnet-4-20250514

# Optional: Override default temperature
# PROMPT_OPT_DEFAULT_TEMPERATURE=0.0

# Optional: Override max concurrent API requests
# PROMPT_OPT_MAX_CONCURRENT_REQUESTS=10

# Optional: Override datasets directory
# PROMPT_OPT_DATASETS_DIR=datasets

# Optional: Override runs directory
# PROMPT_OPT_RUNS_DIR=runs
```

---

### 17. `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
*.egg
dist/
build/

# Virtual environments
.venv/
venv/
env/

# Environment variables (secrets)
.env

# Prompt Optimizer data
runs/
datasets/*/splits/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Ruff
.ruff_cache/
```

---

## Testing Requirements

After all files are created, the following commands must succeed:

### 1. Install the project

```bash
cd prompt-optimizer
uv sync
```

This installs all runtime and dev dependencies and registers the `prompt-opt` script.

### 2. Pytest passes

```bash
uv run pytest
```

Expected: passes with 0 tests collected (or any auto-discovered tests pass). Exit code 0. If pytest fails because zero tests are collected, add the following to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
filterwarnings = [
    "ignore::DeprecationWarning",
]
# Allow zero tests to pass without error
minversion = "8.0"
```

Note: pytest with `testpaths = ["tests"]` and no test files/functions may return exit code 5 ("no tests collected"). To handle this, create a minimal passing test file:

#### `tests/test_scaffold.py`

```python
"""Smoke tests verifying the project scaffold is set up correctly."""

from __future__ import annotations

from prompt_optimizer import __version__
from prompt_optimizer.config import Settings, get_settings


def test_version_is_set() -> None:
    """Verify the package version is defined."""
    assert __version__ == "0.1.0"


def test_settings_defaults() -> None:
    """Verify Settings can be instantiated with default values."""
    settings = Settings(anthropic_api_key="test-key")
    assert settings.default_model == "claude-sonnet-4-20250514"
    assert settings.default_temperature == 0.0
    assert settings.max_concurrent_requests == 10


def test_settings_require_api_key_raises() -> None:
    """Verify require_api_key raises when key is empty."""
    settings = Settings()
    try:
        settings.require_api_key()
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_settings_require_api_key_returns_key() -> None:
    """Verify require_api_key returns key when set."""
    settings = Settings(anthropic_api_key="sk-test-123")
    assert settings.require_api_key() == "sk-test-123"


def test_mock_settings_fixture(mock_settings: Settings) -> None:
    """Verify the mock_settings fixture provides usable settings."""
    assert mock_settings.anthropic_api_key == "test-api-key-not-real"
    assert mock_settings.datasets_dir.exists()
    assert mock_settings.runs_dir.exists()
```

### 3. Ruff passes

```bash
uv run ruff check .
```

Expected: exit code 0, no errors.

### 4. CLI help works

```bash
uv run prompt-opt --help
```

Expected output includes the group description and lists all subcommands: `init`, `add`, `import`, `split`, `config`, `run`, `status`, `results`, `report`, `export`.

### 5. Import works

```bash
uv run python -c "from prompt_optimizer.config import Settings; print('OK')"
```

Expected: prints `OK`.

### 6. Settings loads from environment

```bash
PROMPT_OPT_ANTHROPIC_API_KEY=test123 uv run python -c "
from prompt_optimizer.config import Settings
s = Settings()
assert s.anthropic_api_key == 'test123', f'Got: {s.anthropic_api_key}'
print('Environment loading OK')
"
```

Expected: prints `Environment loading OK`.

---

## Implementation Notes

1. **Use `src/` layout** — all package code goes under `src/prompt_optimizer/`, not `prompt_optimizer/` at the repo root. This is enforced by the `[tool.hatch.build.targets.wheel]` config in pyproject.toml.

2. **Every module gets a docstring** — the docstring should explain what the module *will* contain (for future ticket implementers), not just say "placeholder".

3. **Click, not Typer** — use Click groups for the CLI. Click gives more explicit control over argument parsing and avoids the "magic" type-inference behaviors of Typer.

4. **Lazy API key validation** — the `Settings` class allows `anthropic_api_key` to be empty at construction. The `require_api_key()` method is called explicitly before making API calls. This means `prompt-opt --help`, `prompt-opt config --show`, and other non-API commands work without a key.

5. **`pydantic-settings` is a separate package** — since Pydantic v2, `BaseSettings` lives in the `pydantic-settings` package. Make sure both `pydantic>=2.0` and `pydantic-settings>=2.0` are in the dependencies.

6. **Rich console** — all CLI output uses `rich.console.Console()` for consistent formatting. Define a module-level `console = Console()` in `cli.py` and use `console.print()` instead of `click.echo()`.

7. **`lru_cache` on `get_settings()`** — the settings singleton is cached via `@lru_cache(maxsize=1)` so it's only constructed once per process. Tests can patch `get_settings` to inject test settings.

8. **File creation order** — create directories first (`src/prompt_optimizer/providers/`, `tests/`), then files. Ensure all `__init__.py` files exist before running any imports.

---

## Dependencies on Other Tickets

- **None** — this is Ticket 1, the very first ticket.
- **All subsequent tickets depend on this one.** The project structure, configuration system, and CLI skeleton defined here are used by every other ticket.

---

## Verification Checklist

Run these commands in order after implementation. Every one must succeed:

```bash
# 1. Navigate to project root
cd prompt-optimizer

# 2. Install dependencies
uv sync

# 3. Verify pytest passes
uv run pytest -v

# 4. Verify ruff passes
uv run ruff check .

# 5. Verify CLI entry point
uv run prompt-opt --help

# 6. Verify all subcommands are listed
uv run prompt-opt init --help
uv run prompt-opt add --help
uv run prompt-opt import --help
uv run prompt-opt split --help
uv run prompt-opt config --help
uv run prompt-opt run --help
uv run prompt-opt status --help
uv run prompt-opt results --help
uv run prompt-opt report --help
uv run prompt-opt export --help

# 7. Verify imports
uv run python -c "from prompt_optimizer import __version__; print(__version__)"
uv run python -c "from prompt_optimizer.config import Settings, get_settings; print('OK')"
uv run python -c "from prompt_optimizer.cli import cli; print('OK')"
uv run python -c "from prompt_optimizer.dataset import *; print('OK')"
uv run python -c "from prompt_optimizer.candidate import *; print('OK')"
uv run python -c "from prompt_optimizer.runner import *; print('OK')"
uv run python -c "from prompt_optimizer.judge import *; print('OK')"
uv run python -c "from prompt_optimizer.optimizer import *; print('OK')"
uv run python -c "from prompt_optimizer.controller import *; print('OK')"
uv run python -c "from prompt_optimizer.history import *; print('OK')"
uv run python -c "from prompt_optimizer.providers.anthropic import *; print('OK')"

# 8. Verify environment variable loading
PROMPT_OPT_ANTHROPIC_API_KEY=test123 uv run python -c "
from prompt_optimizer.config import Settings
s = Settings()
assert s.anthropic_api_key == 'test123'
print('Env loading OK')
"
```

If all commands exit with code 0, this ticket is complete.
