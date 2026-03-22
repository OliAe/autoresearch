# Ticket 3: Candidate Representation

## Overview

A "candidate" is the thing being optimized: a frozen, immutable configuration of model + system prompt + parameters. Each optimization round, the Optimizer proposes a new candidate. Candidates are never modified once created — each round creates a new one, with a reference to its parent.

This ticket defines how candidates are represented, created, stored, validated, compared, and versioned.

---

## Acceptance Criteria

- [ ] Pydantic model for `Candidate` with all fields (frozen/immutable)
- [ ] YAML serialization and deserialization (round-trip safe)
- [ ] Auto-generated candidate IDs: `candidate_001`, `candidate_002`, etc. (monotonically increasing per run)
- [ ] Validation: model must be in task's `allowed_models` list
- [ ] Validation: temperature between 0.0 and 1.0 inclusive
- [ ] Validation: max_tokens between 1 and 8192 inclusive
- [ ] Validation: system_prompt non-empty and max 10,000 characters
- [ ] Validation: few_shot_example_ids must reference valid example IDs in the dataset
- [ ] Baseline candidate generator: given a `TaskConfig`, generates a minimal starting candidate
- [ ] `CandidateDiff`: shows what changed between two candidates (for logging and display)
- [ ] Candidate storage: saved as YAML files to `runs/<run_id>/candidates/`
- [ ] Comparison display: rich-formatted side-by-side diff in terminal
- [ ] Immutability enforced (Pydantic `frozen=True` — cannot be modified after creation)

---

## Data Model

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime
from typing import Any


class Candidate(BaseModel):
    """A frozen, immutable LLM program configuration.

    Represents everything needed to make an LLM API call:
    which model, what system prompt, what parameters, and
    optionally which few-shot examples to include.

    Candidates are created once and never modified. Each optimization
    round creates a NEW candidate (potentially based on a parent).
    """
    model_config = ConfigDict(frozen=True)

    id: str                              # e.g., "candidate_001"
    model: str                           # e.g., "claude-sonnet-4-20250514"
    system_prompt: str                   # The full system prompt text
    temperature: float = 0.0            # 0.0 to 1.0
    max_tokens: int = 1024              # 1 to 8192
    few_shot_example_ids: list[str] = [] # List of example IDs from the dataset to include as few-shot
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_id: str | None = None        # ID of the candidate this was derived from (None for baseline)
    reasoning: str = ""                 # Optimizer's reasoning for why this candidate was proposed

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if not 1 <= v <= 8192:
            raise ValueError(f"max_tokens must be between 1 and 8192, got {v}")
        return v

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("system_prompt must not be empty")
        if len(v) > 10000:
            raise ValueError(f"system_prompt must be <= 10000 chars, got {len(v)}")
        return v
```

---

## Candidate Factory

```python
from pathlib import Path


class CandidateFactory:
    """Creates Candidate instances with proper ID generation and validation.

    The factory ensures:
    - IDs are monotonically increasing within a run
    - Models are in the allowed list
    - Few-shot example IDs exist in the dataset
    - Baseline candidates are auto-generated from the task config
    """

    def __init__(self, task_config: TaskConfig, run_dir: Path | None = None):
        self.task_config = task_config
        self.run_dir = run_dir
        self._counter = 0  # Incremented for each candidate created

    def create_baseline(self) -> Candidate:
        """Generate a minimal starting candidate from the task config.

        The baseline gives the optimizer a starting point to improve from.
        It uses:
        - The first allowed model
        - A generic system prompt derived from the task description
        - Temperature 0.0 (deterministic)
        - Max tokens 1024
        - No few-shot examples
        """
        self._counter += 1
        model = self.task_config.allowed_models[0]

        if self.task_config.description:
            system_prompt = (
                f"You are a helpful assistant.\n\n"
                f"Task: {self.task_config.description}\n\n"
                f"Respond concisely and accurately."
            )
        else:
            system_prompt = (
                f"You are a helpful assistant for the task: {self.task_config.name}.\n\n"
                f"Respond concisely and accurately."
            )

        return Candidate(
            id=f"candidate_{self._counter:03d}",
            model=model,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1024,
            few_shot_example_ids=[],
            parent_id=None,
            reasoning="Baseline candidate auto-generated from task config.",
        )

    def create_from_proposal(
        self,
        proposal: "OptimizerProposal",
        parent: Candidate,
    ) -> Candidate:
        """Create a new candidate from an optimizer's proposal.

        Validates all fields against the task config constraints.
        Sets parent_id to the parent's ID for lineage tracking.

        Raises ValueError if the proposed model is not in allowed_models.
        """
        self._counter += 1

        if proposal.model not in self.task_config.allowed_models:
            raise ValueError(
                f"Proposed model '{proposal.model}' not in allowed models: "
                f"{self.task_config.allowed_models}"
            )

        return Candidate(
            id=f"candidate_{self._counter:03d}",
            model=proposal.model,
            system_prompt=proposal.system_prompt,
            temperature=proposal.temperature,
            max_tokens=proposal.max_tokens,
            few_shot_example_ids=proposal.few_shot_example_ids,
            parent_id=parent.id,
            reasoning=proposal.reasoning,
        )

    def next_id(self) -> str:
        """Preview the next candidate ID without incrementing."""
        return f"candidate_{self._counter + 1:03d}"
```

---

## Candidate Diff

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import difflib


class CandidateDiff:
    """Computes and displays differences between two candidates.

    Used by the Controller to show what the Optimizer changed each round.
    """

    @staticmethod
    def diff(a: Candidate, b: Candidate) -> dict[str, tuple[Any, Any]]:
        """Return a dict of changed fields: {field: (old_value, new_value)}.

        Only includes fields that are actually different.
        Excludes: id, created_at, parent_id, reasoning (always different).
        """
        compare_fields = ["model", "system_prompt", "temperature", "max_tokens", "few_shot_example_ids"]
        changes = {}
        for field in compare_fields:
            old = getattr(a, field)
            new = getattr(b, field)
            if old != new:
                changes[field] = (old, new)
        return changes

    @staticmethod
    def display(a: Candidate, b: Candidate, console: Console | None = None) -> None:
        """Rich-formatted display showing what changed between two candidates.

        For system_prompt changes, shows a unified diff.
        For other fields, shows old → new.
        """
        if console is None:
            console = Console()

        changes = CandidateDiff.diff(a, b)

        if not changes:
            console.print("[dim]No changes between candidates.[/dim]")
            return

        console.print(f"\n[bold]Changes: {a.id} → {b.id}[/bold]")

        for field, (old, new) in changes.items():
            if field == "system_prompt":
                # Show unified diff for prompts
                old_lines = old.splitlines(keepends=True)
                new_lines = new.splitlines(keepends=True)
                diff_lines = difflib.unified_diff(
                    old_lines, new_lines,
                    fromfile=a.id, tofile=b.id, lineterm=""
                )
                diff_text = "".join(diff_lines)
                if diff_text:
                    console.print(Panel(diff_text, title="System Prompt Diff"))
            else:
                console.print(f"  {field}: [red]{old}[/red] → [green]{new}[/green]")

        if b.reasoning:
            console.print(f"\n[dim]Reasoning: {b.reasoning}[/dim]")
```

---

## YAML Storage Format

Candidates are saved as YAML files. Example:

```yaml
# runs/run_20260322_143000/candidates/candidate_003.yaml
id: candidate_003
model: claude-sonnet-4-20250514
system_prompt: |
  You are a customer support ticket classifier.

  Classify each ticket into exactly ONE of these categories:
  - billing
  - product_damage
  - shipping
  - account
  - other

  Rules:
  1. Respond with ONLY the category name
  2. No explanation, no punctuation, no capitalization
  3. If unclear, choose the most likely category
temperature: 0.0
max_tokens: 50
few_shot_example_ids: []
created_at: "2026-03-22T14:30:00Z"
parent_id: candidate_002
reasoning: |
  The previous prompt failed on 12 examples because the model added explanation
  text after the category name. Added explicit "respond with ONLY the category
  name" instruction and "no explanation" rule.
```

### Serialization/Deserialization

```python
import yaml
from pathlib import Path


def save_candidate(candidate: Candidate, path: Path) -> None:
    """Save a candidate to a YAML file."""
    data = candidate.model_dump()
    data["created_at"] = data["created_at"].isoformat()
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_candidate(path: Path) -> Candidate:
    """Load a candidate from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Candidate(**data)
```

---

## Baseline Generation Logic

The baseline is intentionally minimal — it gives the optimizer a clear starting point to improve from. It should NOT try to be clever.

| Field | Baseline Value |
|-------|---------------|
| model | First model in `task_config.allowed_models` |
| system_prompt | `"You are a helpful assistant.\n\nTask: {description}\n\nRespond concisely and accurately."` |
| temperature | 0.0 |
| max_tokens | 1024 |
| few_shot_example_ids | [] (empty) |
| parent_id | None |
| reasoning | "Baseline candidate auto-generated from task config." |

The baseline is designed to be mediocre on purpose — the optimizer's job is to improve it.

---

## Testing Requirements

```
# Candidate creation and validation
test_candidate_create_valid
test_candidate_immutable_cannot_modify
test_candidate_invalid_temperature_below_zero
test_candidate_invalid_temperature_above_one
test_candidate_invalid_max_tokens_zero
test_candidate_invalid_max_tokens_too_high
test_candidate_empty_system_prompt_rejected
test_candidate_system_prompt_too_long_rejected
test_candidate_valid_boundary_temperature_zero
test_candidate_valid_boundary_temperature_one

# YAML serialization
test_candidate_yaml_save
test_candidate_yaml_load
test_candidate_yaml_roundtrip  # save → load → compare == original

# Baseline generation
test_baseline_with_description
test_baseline_without_description
test_baseline_uses_first_allowed_model
test_baseline_temperature_is_zero
test_baseline_no_few_shot

# Factory
test_factory_ids_monotonically_increase
test_factory_create_from_proposal
test_factory_create_from_proposal_invalid_model_rejected
test_factory_sets_parent_id

# Diff
test_diff_no_changes
test_diff_prompt_change
test_diff_model_change
test_diff_temperature_change
test_diff_multiple_changes
test_diff_display_renders  # just check it doesn't crash
```

**Minimum: 20 test cases.**

---

## Dependencies

- **Depends on:** Ticket 1 (project scaffolding), Ticket 2 (TaskConfig, Example for few-shot IDs)
- **Depended on by:** Ticket 4 (Runner uses candidates), Ticket 6 (Optimizer creates proposals → factory creates candidates), Ticket 7 (Controller manages candidates)
