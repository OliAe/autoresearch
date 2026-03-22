# Ticket 2: Dataset Store & Input/Output Management

## Overview

The dataset store is where the user puts their input/output examples — the foundation of the entire system. It manages eval examples (input/output pairs) that the optimization loop runs against. Users need to easily add examples one at a time, bulk import from CSV/JSONL, and split them into train/validation/test sets.

The store uses JSONL files on disk — no database needed. Simple, portable, version-controllable.

---

## Acceptance Criteria

- [ ] `prompt-opt init <task_name>` creates a new task directory with `config.yaml` and empty `examples.jsonl`
- [ ] `prompt-opt init <task_name> --description "..."` sets the task description
- [ ] `prompt-opt init <task_name> --models "claude-sonnet-4-20250514,claude-haiku-4-5-20251001"` sets allowed models
- [ ] `prompt-opt add <task_name> --input "..." --output "..."` appends one example to `examples.jsonl`
- [ ] `prompt-opt add <task_name> --input-file input.txt --output-file output.txt` adds from files (for long inputs/outputs)
- [ ] `prompt-opt add <task_name> --id "custom_001"` allows custom IDs
- [ ] Auto-generates IDs if not provided: `ex_001`, `ex_002`, etc. (padded, monotonically increasing)
- [ ] Duplicate detection: warns if an identical input already exists (but still adds it)
- [ ] `prompt-opt import <task_name> --file examples.csv` bulk imports from CSV
- [ ] `prompt-opt import <task_name> --file examples.jsonl` bulk imports from JSONL
- [ ] CSV column names configurable: `--input-column`, `--output-column`, `--id-column`
- [ ] Shows progress bar during bulk import
- [ ] Shows summary after import: "Imported 150 examples (3 duplicates warned)"
- [ ] `prompt-opt split <task_name>` auto-splits into train (70%), validation (15%), test (15%)
- [ ] `prompt-opt split <task_name> --train 0.8 --val 0.1 --test 0.1` custom ratios
- [ ] `prompt-opt split <task_name> --seed 123` deterministic splitting
- [ ] Validates ratios sum to 1.0 (with small float tolerance)
- [ ] `prompt-opt list` shows all tasks in a table
- [ ] `prompt-opt list <task_name>` shows task details (example count, split status, config)
- [ ] `prompt-opt show <task_name> --example-id ex_001` displays a single example
- [ ] `prompt-opt show <task_name> --random --count 5` shows random sample
- [ ] Validation: input and output must be non-empty strings
- [ ] Config.yaml stores task metadata: name, description, created_at, allowed_models, scoring config
- [ ] Test set isolation: log a warning if test split is accessed outside of final evaluation

---

## Data Schemas (Pydantic Models)

### Example

```python
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


class CheckConfig(BaseModel):
    """Configuration for hard checks on a single example.

    These are deterministic checks that the Judge runs before (optionally)
    falling back to LLM-based grading. If no checks are configured, the
    Judge relies entirely on the LLM grader.
    """
    exact_match: bool = False          # Must actual == expected (after strip)
    case_sensitive: bool = True        # Whether exact_match is case-sensitive
    must_contain: list[str] = []       # Phrases that must appear in output
    must_not_contain: list[str] = []   # Phrases that must NOT appear in output
    regex_pattern: str | None = None   # Regex the output must match
    custom_check: str | None = None    # Dotted path to a Python function: "my_module.check_fn"


class Example(BaseModel):
    """A single input/output eval example.

    This is the atomic unit of the dataset. Each example has an input
    (what gets sent to the model) and an expected output (what the model
    should produce). Optionally includes metadata and check configuration.
    """
    id: str                                    # Unique ID: "ex_001", "ex_002", or custom
    input: str                                 # The input text sent to the model
    expected_output: str                       # What the model should produce
    metadata: dict[str, Any] = {}              # Optional tags: {"difficulty": "hard", "category": "billing"}
    checks: CheckConfig = Field(default_factory=CheckConfig)

    def model_post_init(self, __context: Any) -> None:
        if not self.input.strip():
            raise ValueError("input must be a non-empty string")
        if not self.expected_output.strip():
            raise ValueError("expected_output must be a non-empty string")
```

### Task Config

```python
class ScoringConfig(BaseModel):
    """Scoring weights and thresholds for the Judge.

    These are HUMAN-DEFINED and LOCKED — the optimizer cannot change them.
    """
    hard_check_weight: float = 0.7     # Weight for deterministic checks
    llm_grade_weight: float = 0.3      # Weight for LLM grading
    pass_threshold: float = 0.8        # Score >= this = "passed"
    grading_model: str = "claude-sonnet-4-20250514"  # Model used for LLM grading
    grading_prompt: str | None = None  # Custom grading prompt (None = use default)

    def model_post_init(self, __context: Any) -> None:
        if abs((self.hard_check_weight + self.llm_grade_weight) - 1.0) > 0.01:
            raise ValueError("hard_check_weight + llm_grade_weight must equal 1.0")


class TaskConfig(BaseModel):
    """Configuration for a single optimization task.

    Created by `prompt-opt init` and stored as config.yaml in the task directory.
    """
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    allowed_models: list[str] = Field(default_factory=lambda: ["claude-sonnet-4-20250514"])
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
```

### Dataset Class

```python
import random
from pathlib import Path


class Dataset:
    """Manages loading and accessing examples for a task.

    All data lives on disk as JSONL files. The Dataset class provides
    methods to add, import, split, and query examples. It never holds
    all examples in memory unnecessarily.
    """

    def __init__(self, task_dir: Path):
        """Initialize with the path to a task directory (e.g., datasets/my_task/)."""
        self.task_dir = task_dir
        self.examples_path = task_dir / "examples.jsonl"
        self.config_path = task_dir / "config.yaml"
        self.splits_dir = task_dir / "splits"

    def exists(self) -> bool:
        """Check if this task directory exists and is valid."""
        return self.config_path.exists() and self.examples_path.exists()

    def load_config(self) -> TaskConfig:
        """Load and return the task configuration."""
        ...

    def save_config(self, config: TaskConfig) -> None:
        """Save task configuration to config.yaml."""
        ...

    def load_all(self) -> list[Example]:
        """Load all examples from examples.jsonl."""
        ...

    def load_split(self, split: str) -> list[Example]:
        """Load examples for a specific split ('train', 'validation', 'test').

        The split files contain only example IDs. This method reads the IDs,
        then loads the corresponding examples from examples.jsonl.

        Raises FileNotFoundError if splits haven't been created yet.
        Logs a warning if split == 'test' (test set should only be used for final eval).
        """
        ...

    def add_example(self, example: Example) -> None:
        """Append a single example to examples.jsonl.

        Checks for duplicate inputs and warns (but still adds).
        """
        ...

    def import_csv(self, file_path: Path, input_col: str = "input",
                   output_col: str = "output", id_col: str | None = None) -> int:
        """Bulk import examples from a CSV file.

        Returns the number of examples imported.
        Raises ValueError if required columns are missing.
        """
        ...

    def import_jsonl(self, file_path: Path) -> int:
        """Bulk import examples from a JSONL file.

        Each line must be a JSON object with at least 'input' and 'output' fields.
        Returns the number of examples imported.
        """
        ...

    def split(self, train: float = 0.7, val: float = 0.15,
              test: float = 0.15, seed: int = 42) -> dict[str, int]:
        """Split examples into train/validation/test sets.

        Creates splits/ directory with train.jsonl, validation.jsonl, test.jsonl.
        Each split file contains one example ID per line (JSON strings).

        Returns {"train": count, "validation": count, "test": count}.

        Raises ValueError if ratios don't sum to 1.0.
        Warns if total examples < 10.
        """
        ...

    def get_example_by_id(self, example_id: str) -> Example | None:
        """Look up a single example by ID. Returns None if not found."""
        ...

    def get_random_examples(self, count: int = 5) -> list[Example]:
        """Return a random sample of examples."""
        ...

    def next_id(self) -> str:
        """Generate the next auto-incremented example ID.

        Reads existing examples to find the highest numeric ID,
        then returns the next one: ex_001, ex_002, etc.
        """
        ...

    @property
    def total_count(self) -> int:
        """Total number of examples."""
        ...

    @property
    def has_splits(self) -> bool:
        """Whether the dataset has been split."""
        ...

    @property
    def split_counts(self) -> dict[str, int] | None:
        """Counts per split, or None if not split yet."""
        ...

    @property
    def stats(self) -> dict:
        """Summary statistics: total count, split counts, avg input/output length, etc."""
        ...
```

---

## File Layout on Disk

```
datasets/
  my_task/
    config.yaml              # Task metadata, scoring config, allowed models
    examples.jsonl           # Master list of ALL examples (append-only)
    splits/
      train.jsonl            # Example IDs only (references into examples.jsonl)
      validation.jsonl       # Example IDs only
      test.jsonl             # Example IDs only
```

### Why split files contain only IDs

The split files contain just the example IDs, NOT copies of the full examples. This means:
- No data duplication
- Re-splitting is cheap (just rewrite the ID files)
- Editing an example in examples.jsonl automatically updates all splits
- The source of truth is always examples.jsonl

**Split file format** (each line is a JSON string):
```jsonl
"ex_001"
"ex_005"
"ex_012"
```

### config.yaml format
```yaml
name: my_classifier
description: "Classify customer support tickets into categories"
created_at: "2026-03-22T14:00:00Z"
updated_at: "2026-03-22T14:30:00Z"
allowed_models:
  - claude-sonnet-4-20250514
scoring:
  hard_check_weight: 0.7
  llm_grade_weight: 0.3
  pass_threshold: 0.8
  grading_model: claude-sonnet-4-20250514
  grading_prompt: null
```

### examples.jsonl format
```jsonl
{"id": "ex_001", "input": "My order arrived broken", "expected_output": "product_damage", "metadata": {"category": "easy"}, "checks": {"exact_match": true, "case_sensitive": false}}
{"id": "ex_002", "input": "I was charged twice for my subscription", "expected_output": "billing", "metadata": {}, "checks": {"exact_match": true, "case_sensitive": false}}
{"id": "ex_003", "input": "When will my package arrive?", "expected_output": "shipping", "metadata": {}, "checks": {"exact_match": true, "case_sensitive": false}}
```

---

## CLI Command Details

### `prompt-opt init <task_name>`

**Signature:**
```
prompt-opt init <task_name> [--description TEXT] [--models TEXT] [--force]
```

**Behavior:**
1. Check if `datasets/<task_name>/` already exists. If yes, error (unless `--force`).
2. Create directory `datasets/<task_name>/`
3. Create `config.yaml` with task metadata
4. Create empty `examples.jsonl`
5. Display confirmation panel

**Terminal output:**
```
╭─── Task Created ─────────────────────────────────────╮
│ Name:        my_classifier                           │
│ Directory:   datasets/my_classifier/                 │
│ Description: Classify customer support tickets       │
│ Models:      claude-sonnet-4-20250514                │
╰──────────────────────────────────────────────────────╯

Next steps:
  1. Add examples:  prompt-opt add my_classifier --input "..." --output "..."
  2. Bulk import:   prompt-opt import my_classifier --file examples.csv
  3. Split dataset: prompt-opt split my_classifier
  4. Optimize:      prompt-opt run my_classifier
```

**Error cases:**
- Task already exists: `"Error: Task 'my_classifier' already exists at datasets/my_classifier/. Use --force to overwrite."`

---

### `prompt-opt add <task_name>`

**Signature:**
```
prompt-opt add <task_name> --input TEXT --output TEXT [--id TEXT] [--metadata JSON]
prompt-opt add <task_name> --input-file PATH --output-file PATH [--id TEXT]
```

**Behavior:**
1. Validate task exists
2. Read input (from `--input` flag or `--input-file`)
3. Read output (from `--output` flag or `--output-file`)
4. Generate ID if not provided
5. Check for duplicate inputs (warn but don't block)
6. Append to `examples.jsonl`
7. Display the added example

**Terminal output:**
```
╭─── Example Added ────────────────────────────────────╮
│ ID:     ex_004                                       │
│ Input:  "My order is 3 weeks late"                   │
│ Output: "shipping"                                   │
╰──────────────────────────────────────────────────────╯
Total examples: 4
```

**Duplicate warning:**
```
⚠ Warning: An example with this input already exists (ex_002). Adding anyway.
```

**Error cases:**
- Task doesn't exist: `"Error: Task 'foo' not found. Run 'prompt-opt init foo' first."`
- Empty input: `"Error: Input must be non-empty."`
- Empty output: `"Error: Output must be non-empty."`
- File not found: `"Error: File not found: /path/to/input.txt"`

---

### `prompt-opt import <task_name>`

**Signature:**
```
prompt-opt import <task_name> --file PATH [--input-column TEXT] [--output-column TEXT] [--id-column TEXT]
```

**Behavior:**
1. Validate task exists
2. Detect file format from extension (.csv or .jsonl)
3. Validate required columns/fields exist
4. Show progress bar during import
5. Skip invalid rows with warnings
6. Show summary

**Terminal output:**
```
Importing from examples.csv...
[████████████████████████████████] 153/153 rows

╭─── Import Complete ──────────────────────────────────╮
│ Imported:  150 examples                              │
│ Skipped:   3 (empty input or output)                 │
│ Duplicates: 5 (warned, still imported)               │
│ Total:     154 examples in dataset                   │
╰──────────────────────────────────────────────────────╯
```

**Error cases:**
- Unsupported format: `"Error: Unsupported file format '.xlsx'. Use .csv or .jsonl."`
- Missing columns: `"Error: CSV missing required column 'output'. Found columns: ['input', 'label', 'category']. Use --output-column to specify."`
- File not found: `"Error: File not found: examples.csv"`

---

### `prompt-opt split <task_name>`

**Signature:**
```
prompt-opt split <task_name> [--train FLOAT] [--val FLOAT] [--test FLOAT] [--seed INT]
```

**Defaults:** `--train 0.7 --val 0.15 --test 0.15 --seed 42`

**Behavior:**
1. Validate ratios sum to 1.0
2. Load all example IDs
3. Shuffle with seed
4. Split into groups
5. Write split files to `splits/` directory
6. Show results table

**Terminal output:**
```
╭─── Dataset Split ────────────────────────────────────╮
│                                                      │
│  Split        Count    Percentage                    │
│  ─────────────────────────────────                   │
│  Train        140      70.0%                         │
│  Validation   30       15.0%                         │
│  Test         30       15.0%                         │
│  ─────────────────────────────────                   │
│  Total        200      100.0%                        │
│                                                      │
│  Seed: 42                                            │
╰──────────────────────────────────────────────────────╯
```

**Warnings:**
- `< 10 examples: "⚠ Warning: Only 8 examples. Splits may be too small for reliable optimization."`
- Re-splitting: `"Splits already exist. Overwrite? [y/N]:"`

**Error cases:**
- Ratios don't sum to 1.0: `"Error: Split ratios must sum to 1.0. Got: 0.6 + 0.2 + 0.1 = 0.9"`
- No examples: `"Error: No examples in dataset. Add some first with 'prompt-opt add'."`

---

### `prompt-opt list`

**Signature:**
```
prompt-opt list                    # list all tasks
prompt-opt list <task_name>        # show task details
```

**No args — table of all tasks:**
```
╭─── Tasks ────────────────────────────────────────────╮
│                                                      │
│  Task              Examples    Split?    Created      │
│  ─────────────────────────────────────────────────    │
│  my_classifier     200         ✓         2026-03-22  │
│  email_summarizer  50          ✗         2026-03-21  │
│  code_reviewer     0           ✗         2026-03-20  │
│                                                      │
╰──────────────────────────────────────────────────────╯
```

**With task name — detailed view:**
```
╭─── Task: my_classifier ─────────────────────────────────────╮
│                                                             │
│  Description:  Classify customer support tickets            │
│  Created:      2026-03-22 14:00:00                          │
│  Updated:      2026-03-22 14:30:00                          │
│  Examples:     200                                          │
│                                                             │
│  Splits:                                                    │
│    Train:      140 (70%)                                    │
│    Validation: 30 (15%)                                     │
│    Test:       30 (15%)                                     │
│                                                             │
│  Allowed Models:                                            │
│    - claude-sonnet-4-20250514                               │
│                                                             │
│  Scoring:                                                   │
│    Hard check weight: 0.7                                   │
│    LLM grade weight:  0.3                                   │
│    Pass threshold:    0.8                                   │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
```

---

## Edge Cases to Handle

| Scenario | Behavior |
|----------|----------|
| `add` before `init` | Error: "Task 'x' not found. Run `prompt-opt init x` first." |
| CSV with missing columns | Error listing which columns are missing and which were found |
| Split with < 10 examples | Warning but proceed |
| Re-split existing splits | Confirmation prompt, then overwrite |
| Unicode/multiline inputs | Must handle correctly — JSONL supports this natively |
| Very long inputs (>10K chars) | Warn but allow: "⚠ Example ex_042 input is 15,234 chars. This may use many tokens." |
| Empty input or output | Reject with clear error |
| Duplicate example IDs in import | Reject the duplicate, warn, continue importing others |
| JSONL with missing fields | Skip the line, warn, continue |
| CSV with extra columns | Ignore extra columns, import normally |
| ID collision (auto-generated ID matches existing) | Auto-increment past the collision |
| Concurrent writes | Not supported in v1 — single-user, single-process |

---

## Testing Requirements

### Unit Tests for Dataset Class
```
test_dataset_init_creates_directory
test_dataset_init_creates_config
test_dataset_init_creates_empty_examples
test_dataset_init_with_description
test_dataset_init_with_models
test_dataset_add_example_basic
test_dataset_add_example_auto_id
test_dataset_add_example_custom_id
test_dataset_add_example_duplicate_input_warns
test_dataset_add_example_empty_input_rejected
test_dataset_add_example_empty_output_rejected
test_dataset_import_csv_basic
test_dataset_import_csv_custom_columns
test_dataset_import_csv_missing_columns_error
test_dataset_import_jsonl_basic
test_dataset_import_jsonl_missing_fields_skipped
test_dataset_split_default_ratios
test_dataset_split_custom_ratios
test_dataset_split_invalid_ratios_error
test_dataset_split_deterministic_with_seed
test_dataset_split_same_seed_same_result
test_dataset_load_split_train
test_dataset_load_split_validation
test_dataset_load_split_test_warns
test_dataset_load_split_not_split_yet_error
test_dataset_get_example_by_id
test_dataset_get_example_by_id_not_found
test_dataset_get_random_examples
test_dataset_next_id_auto_increment
test_dataset_next_id_skips_collisions
test_dataset_stats
test_dataset_total_count
```

### Pydantic Model Tests
```
test_example_valid
test_example_empty_input_rejected
test_example_empty_output_rejected
test_example_default_checks
test_check_config_defaults
test_task_config_defaults
test_task_config_yaml_roundtrip
test_scoring_config_weights_must_sum_to_one
```

### CLI Integration Tests
```
test_cli_init_success
test_cli_init_already_exists_error
test_cli_init_with_force_overwrites
test_cli_add_success
test_cli_add_task_not_found_error
test_cli_add_from_files
test_cli_import_csv
test_cli_import_jsonl
test_cli_import_unsupported_format_error
test_cli_split_default
test_cli_split_custom_ratios
test_cli_split_bad_ratios_error
test_cli_list_all
test_cli_list_single_task
test_cli_show_by_id
test_cli_show_random
```

Use Click's `CliRunner` for CLI tests. Use `tmp_path` fixture for isolated filesystem.

**Minimum: 30 test cases.**

---

## Implementation Notes

- Use Python's `csv` module for CSV import (not pandas — avoid heavy dependency)
- Use standard `json` for JSONL read/write
- JSONL files should be opened in append mode for `add_example`
- For `split`, use `random.shuffle` with a seeded `random.Random` instance (don't touch global state)
- File paths should always go through `pathlib.Path` (no string concatenation)
- All disk I/O should handle `FileNotFoundError` and `PermissionError` gracefully
- The `Dataset` class should be independent of CLI — it's a library class that the CLI and Controller both use

---

## Dependencies

- **Depends on:** Ticket 1 (project scaffolding, config, CLI skeleton, Settings)
- **Depended on by:** Ticket 4 (Runner reads examples), Ticket 5 (Judge reads examples and check configs), Ticket 6 (Optimizer reads task config), Ticket 7 (Controller uses Dataset)
