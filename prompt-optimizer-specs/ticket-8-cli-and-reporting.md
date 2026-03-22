# Ticket 8: CLI Interface & Reporting

## Overview

The CLI is the user's interface to the entire system. All interaction happens through the `prompt-opt` command. This ticket wires all the previously built components into user-facing commands with proper argument parsing, validation, error handling, and Rich-formatted output.

The CLI uses **Click** for commands and **Rich** for terminal output.

---

## Acceptance Criteria

- [ ] `prompt-opt` is a registered CLI entry point (installed via `uv pip install -e .`)
- [ ] Every command has `--help` with usage description
- [ ] Rich-formatted output: tables, panels, progress bars, colored text
- [ ] Global `--verbose` flag enables debug logging
- [ ] Global `--quiet` flag shows only essential output
- [ ] All errors are user-friendly (no stack traces unless `--verbose`)
- [ ] All commands handle missing prerequisites gracefully with actionable error messages
- [ ] Confirmation prompt before starting optimization (shows estimated cost)

---

## Command Reference

### Global Options

```python
@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
@click.pass_context
def cli(ctx, verbose, quiet):
    """Prompt Optimizer — Automated LLM prompt engineering."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["settings"] = Settings()

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)
```

---

### `prompt-opt init <task_name>`

**Signature:**
```
prompt-opt init <task_name> [--description TEXT] [--models TEXT] [--force]
```

**What it does:**
1. Validates task name (alphanumeric + underscores only)
2. Creates `datasets/<task_name>/` directory
3. Creates `config.yaml` with task metadata
4. Creates empty `examples.jsonl`

**Success output:**
```
╭─── Task Created ────────────────────────────────────╮
│ Name:        my_classifier                          │
│ Directory:   datasets/my_classifier/                │
│ Description: Classify support tickets               │
│ Models:      claude-sonnet-4-20250514               │
╰─────────────────────────────────────────────────────╯

Next steps:
  1. Add examples:  prompt-opt add my_classifier --input "..." --output "..."
  2. Bulk import:   prompt-opt import my_classifier --file examples.csv
  3. Split dataset: prompt-opt split my_classifier
  4. Optimize:      prompt-opt run my_classifier
```

**Errors:**
- Already exists: `"Error: Task 'my_classifier' already exists. Use --force to overwrite."`
- Invalid name: `"Error: Task name must be alphanumeric with underscores. Got: 'my-task!'"`

---

### `prompt-opt add <task_name>`

**Signature:**
```
prompt-opt add <task_name> --input TEXT --output TEXT [--id TEXT] [--metadata JSON]
prompt-opt add <task_name> --input-file PATH --output-file PATH [--id TEXT]
```

**Implementation:**
```python
@cli.command()
@click.argument("task_name")
@click.option("--input", "input_text", type=str, help="Input text")
@click.option("--output", "output_text", type=str, help="Expected output text")
@click.option("--input-file", type=click.Path(exists=True), help="Read input from file")
@click.option("--output-file", type=click.Path(exists=True), help="Read output from file")
@click.option("--id", "example_id", type=str, help="Custom example ID")
@click.option("--metadata", type=str, help="JSON metadata string")
@click.pass_context
def add(ctx, task_name, input_text, output_text, input_file, output_file, example_id, metadata):
    """Add a single input/output example to a task."""
    settings = ctx.obj["settings"]
    dataset = Dataset(settings.datasets_dir / task_name)

    if not dataset.exists():
        _error(f"Task '{task_name}' not found. Run `prompt-opt init {task_name}` first.")
        return

    # Read input from flag or file
    if input_file:
        input_text = Path(input_file).read_text().strip()
    if output_file:
        output_text = Path(output_file).read_text().strip()

    if not input_text or not output_text:
        _error("Both --input and --output are required (or use --input-file and --output-file).")
        return

    # Parse metadata
    meta = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            _error("--metadata must be valid JSON.")
            return

    # Generate or validate ID
    if not example_id:
        example_id = dataset.next_id()

    example = Example(id=example_id, input=input_text, expected_output=output_text, metadata=meta)
    dataset.add_example(example)

    console.print(Panel(
        f"ID:     {example.id}\n"
        f"Input:  {_truncate(example.input, 80)}\n"
        f"Output: {_truncate(example.expected_output, 80)}",
        title="Example Added",
        border_style="green",
    ))
    console.print(f"Total examples: {dataset.total_count}")
```

---

### `prompt-opt import <task_name>`

**Signature:**
```
prompt-opt import <task_name> --file PATH [--input-column TEXT] [--output-column TEXT] [--id-column TEXT]
```

**Behavior:**
1. Detect format from file extension (.csv or .jsonl)
2. Validate columns/fields exist
3. Import with progress bar
4. Show summary

**Success output:**
```
Importing from examples.csv...
[████████████████████████████████] 153/153

╭─── Import Complete ─────────────────────────────────╮
│ Imported:   150 examples                            │
│ Skipped:    3 (invalid or empty)                    │
│ Total:      154 examples in dataset                 │
╰─────────────────────────────────────────────────────╯
```

**Errors:**
- Unsupported format: `"Error: Unsupported file format '.xlsx'. Use .csv or .jsonl."`
- Missing columns: `"Error: CSV missing column 'output'. Found: ['input', 'label']. Use --output-column label"`
- File not found: handled by Click's `exists=True`

---

### `prompt-opt split <task_name>`

**Signature:**
```
prompt-opt split <task_name> [--train FLOAT] [--val FLOAT] [--test FLOAT] [--seed INT]
```

**Defaults:** `--train 0.7 --val 0.15 --test 0.15 --seed 42`

**Success output:**
```
╭─── Dataset Split ───────────────────────────────────╮
│                                                     │
│  Split         Count    Percentage                  │
│  ──────────────────────────────                     │
│  Train         140      70.0%                       │
│  Validation    30       15.0%                       │
│  Test          30       15.0%                       │
│  ──────────────────────────────                     │
│  Total         200      100.0%                      │
│  Seed: 42                                           │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

**Validation:**
- Ratios must sum to 1.0: `"Error: Ratios must sum to 1.0. Got 0.6 + 0.2 + 0.1 = 0.9"`
- Warns if < 10 examples
- Confirms before re-splitting existing splits

---

### `prompt-opt list [task_name]`

**Signature:**
```
prompt-opt list                    # All tasks
prompt-opt list <task_name>        # Single task details
```

**All tasks:**
```
╭─── Tasks ───────────────────────────────────────────╮
│                                                     │
│  Task              Examples   Split?   Created      │
│  ────────────────────────────────────────────────    │
│  my_classifier     200        Yes      2026-03-22   │
│  email_summarizer  50         No       2026-03-21   │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

---

### `prompt-opt show <task_name>`

**Signature:**
```
prompt-opt show <task_name> --example-id TEXT
prompt-opt show <task_name> --random [--count INT]
```

**Output:**
```
╭─── Example: ex_001 ─────────────────────────────────╮
│                                                     │
│  Input:                                             │
│    My order arrived and the box was completely       │
│    crushed. The product inside is broken.            │
│                                                     │
│  Expected Output:                                   │
│    product_damage                                   │
│                                                     │
│  Checks:                                            │
│    exact_match: true (case_insensitive)             │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

---

### `prompt-opt config <task_name>`

**Signature:**
```
prompt-opt config <task_name>                              # Show current config
prompt-opt config <task_name> --models "sonnet,haiku"      # Update models
prompt-opt config <task_name> --hard-weight 0.5            # Update scoring
```

---

### `prompt-opt run <task_name>` — THE MAIN COMMAND

**Signature:**
```
prompt-opt run <task_name> [--max-rounds INT] [--target-accuracy FLOAT]
    [--patience INT] [--max-budget FLOAT] [--no-test] [--resume RUN_ID]
```

**Defaults:** `--max-rounds 50 --target-accuracy 0.95 --patience 5 --max-budget 50.0`

**Pre-flight validation (check ALL before starting):**
1. Task exists
2. API key is set
3. Examples exist (> 0)
4. Dataset is split
5. At least 5 train examples and 3 validation examples

**Error messages for each:**
```
Error: ANTHROPIC_API_KEY not set.
  Set it in .env file or as environment variable:
  export ANTHROPIC_API_KEY=sk-ant-...

Error: Task 'foo' has no examples.
  Add examples first: prompt-opt add foo --input "..." --output "..."

Error: Task 'foo' has not been split.
  Run: prompt-opt split foo
```

**Confirmation before starting:**
```
╭─── Prompt Optimizer ───────────────────────────────╮
│                                                    │
│ Task:       my_classifier                          │
│ Examples:   200 (140 train / 30 val / 30 test)     │
│ Models:     claude-sonnet-4-20250514               │
│ Max rounds: 50                                     │
│ Target:     95.0%                                  │
│ Budget:     $50.00                                 │
│                                                    │
╰────────────────────────────────────────────────────╯
Proceed? [Y/n]:
```

**During execution:** Delegates to `Controller.run()` which handles its own progress display.

**After completion:** Show summary and path to full report.

**Implementation:**
```python
@cli.command()
@click.argument("task_name")
@click.option("--max-rounds", type=int, default=50, help="Maximum optimization rounds")
@click.option("--target-accuracy", type=float, default=0.95, help="Target accuracy (0.0-1.0)")
@click.option("--patience", type=int, default=5, help="Rounds before plateau detection")
@click.option("--max-budget", type=float, default=50.0, help="Maximum API spend in USD")
@click.option("--no-test", is_flag=True, help="Skip final test set evaluation")
@click.option("--resume", "resume_run_id", type=str, help="Resume a previous run")
@click.pass_context
def run(ctx, task_name, max_rounds, target_accuracy, patience, max_budget, no_test, resume_run_id):
    """Run the optimization loop for a task."""
    settings = ctx.obj["settings"]

    # Pre-flight validation
    errors = _validate_prerequisites(task_name, settings)
    if errors:
        for error in errors:
            _error(error)
        return

    # Show confirmation
    dataset = Dataset(settings.datasets_dir / task_name)
    if not _confirm_run(dataset, max_rounds, target_accuracy, max_budget):
        console.print("Cancelled.")
        return

    # Run
    controller = Controller(
        task_name=task_name,
        settings=settings,
        max_rounds=max_rounds,
        target_accuracy=target_accuracy,
        patience=patience,
        max_budget_usd=max_budget,
        run_test_eval=not no_test,
    )

    report = asyncio.run(controller.run())

    # Final output
    console.print(f"\nReport saved to: runs/{report.run_id}/report.md")
    console.print(f"Champion config: runs/{report.run_id}/champion.yaml")
```

---

### `prompt-opt status <task_name>`

**Signature:**
```
prompt-opt status <task_name> [--run-id TEXT]
```

Shows status of the latest or specified run. Reads from the run directory.

---

### `prompt-opt results <task_name>`

**Signature:**
```
prompt-opt results <task_name> [--run-id TEXT] [--round INT]
```

**Output:**
```
╭─── Results: my_classifier (run_20260322_143000) ────╮
│                                                     │
│ Round  Val Accuracy  Model    Change       Champion  │
│ ────────────────────────────────────────────────     │
│ 1      65.0%         sonnet   Baseline     Yes      │
│ 2      72.0%         sonnet   Format fix   Yes      │
│ 3      78.0%         sonnet   Category     Yes      │
│ ...                                                 │
│ 23     95.1%         sonnet   Edge cases   Yes      │
│                                                     │
│ Stopping reason: Target accuracy reached            │
│ Total cost: $3.47                                   │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

---

### `prompt-opt report <task_name>`

**Signature:**
```
prompt-opt report <task_name> [--run-id TEXT] [--format markdown|json] [--output PATH]
```

- Default: print markdown to terminal
- `--output report.md`: save to file
- `--format json`: machine-readable output

---

### `prompt-opt export <task_name>`

**Signature:**
```
prompt-opt export <task_name> [--run-id TEXT] [--output PATH]
```

- Exports champion config as standalone YAML
- Default output: `<task_name>_champion.yaml`
- The YAML is self-contained — any developer can use it directly

**Output:**
```
Champion exported to: my_classifier_champion.yaml

Contents:
  Model:       claude-sonnet-4-20250514
  Temperature: 0.0
  Max tokens:  50
  Prompt:      118 characters
```

**Exported YAML format:**
```yaml
# Prompt Optimizer Champion Configuration
# Task: my_classifier
# Run: run_20260322_143000
# Validation accuracy: 95.1%
# Test accuracy: 93.8%

model: claude-sonnet-4-20250514
system_prompt: |
  You are a customer support ticket classifier...
temperature: 0.0
max_tokens: 50
```

---

## Error Handling Strategy

All errors should be displayed in a consistent format:

```python
def _error(message: str) -> None:
    """Display an error message in a red panel."""
    console.print(Panel(
        message,
        title="Error",
        border_style="red",
    ))


def _warning(message: str) -> None:
    """Display a warning message in a yellow panel."""
    console.print(Panel(
        message,
        title="Warning",
        border_style="yellow",
    ))
```

| Scenario | Error Message |
|----------|--------------|
| Missing API key | `ANTHROPIC_API_KEY not set. Set it in .env or export ANTHROPIC_API_KEY=sk-ant-...` |
| Missing task | `Task 'foo' not found. Run 'prompt-opt init foo' first.` |
| No examples | `Task 'foo' has no examples. Add some with 'prompt-opt add foo --input ... --output ...'` |
| No splits | `Task 'foo' has not been split. Run 'prompt-opt split foo'.` |
| API connection error | `API connection failed. Check your internet and API key.` |
| Invalid ratios | `Split ratios must sum to 1.0. Got: 0.6 + 0.2 + 0.1 = 0.9` |

**No raw stack traces** unless `--verbose` is set. Catch exceptions at the command level and display user-friendly messages.

---

## Rich Output Standards

- Use `rich.console.Console()` for ALL output — no raw `print()`
- Use `rich.table.Table` for tabular data
- Use `rich.panel.Panel` for highlighted information
- Use `rich.progress.Progress` for long operations
- Use `rich.syntax.Syntax` for displaying code/prompts with syntax highlighting
- Colors: `green` = success, `red` = error, `yellow` = warning, `blue` = info, `bold` = emphasis
- All panels should have descriptive titles

---

## CLI Registration

In `pyproject.toml`:
```toml
[project.scripts]
prompt-opt = "prompt_optimizer.cli:cli"
```

This means after `uv pip install -e .`, the `prompt-opt` command is available globally.

---

## Testing Requirements

Use Click's `CliRunner` for all CLI tests:

```python
from click.testing import CliRunner
from prompt_optimizer.cli import cli

def test_init_success(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "test_task", "--description", "Test task"])
    assert result.exit_code == 0
    assert "Task Created" in result.output
```

```
# Init
test_cli_init_success
test_cli_init_with_description
test_cli_init_with_models
test_cli_init_already_exists_error
test_cli_init_force_overwrites
test_cli_init_invalid_name

# Add
test_cli_add_success
test_cli_add_task_not_found
test_cli_add_from_files
test_cli_add_missing_input_error
test_cli_add_missing_output_error

# Import
test_cli_import_csv_success
test_cli_import_jsonl_success
test_cli_import_unsupported_format
test_cli_import_missing_columns

# Split
test_cli_split_default_ratios
test_cli_split_custom_ratios
test_cli_split_invalid_ratios
test_cli_split_no_examples

# List
test_cli_list_all_tasks
test_cli_list_single_task
test_cli_list_no_tasks

# Show
test_cli_show_by_id
test_cli_show_random
test_cli_show_not_found

# Config
test_cli_config_show
test_cli_config_update_models

# Run
test_cli_run_validates_prerequisites
test_cli_run_missing_api_key
test_cli_run_missing_splits
test_cli_run_starts_controller (mock Controller)

# Results
test_cli_results_shows_table
test_cli_results_no_runs

# Report
test_cli_report_markdown
test_cli_report_json

# Export
test_cli_export_success
test_cli_export_no_runs

# Global flags
test_cli_verbose_enables_debug
test_cli_quiet_minimal_output
test_cli_help_shows_all_commands
```

**Minimum: 30 test cases.**

---

## Implementation Notes

- Use `click.group()` for the main CLI, `click.command()` for subcommands
- Use `@click.pass_context` and `ctx.obj` to pass Settings through the command chain
- Handle `KeyboardInterrupt` gracefully in the `run` command
- The `run` command uses `asyncio.run()` to execute the async Controller
- All file paths use `pathlib.Path` — never string concatenation
- The `import` command name conflicts with Python's `import` keyword — use `import_data` as the function name with `@cli.command("import")` decorator
- The CLI should work immediately after `uv pip install -e .` with no additional setup beyond setting the API key

---

## Dependencies

- **Depends on:** ALL previous tickets (1-7)
- **This is the final ticket** — nothing depends on it
- After this ticket is complete, the entire system is functional end-to-end
