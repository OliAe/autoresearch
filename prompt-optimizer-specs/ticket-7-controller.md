# Ticket 7: Controller (Main Loop Orchestration)

## Overview

The Controller is the glue that ties everything together. It runs the optimization loop: each round calls the Runner → Judge → Optimizer in sequence, manages champion/challenger evaluation, tracks history, enforces stopping criteria, and produces a final report.

This is the component the CLI calls to start an optimization run.

---

## Acceptance Criteria

- [ ] `Controller.run() -> OptimizationReport` runs the full optimization loop
- [ ] Creates baseline candidate automatically on first round (via CandidateFactory)
- [ ] Each round: run candidate on train → score → analyze → propose challenger → run challenger on validation → promote if better
- [ ] Champion tracking: best candidate based on VALIDATION set score (not train set)
- [ ] Challenger promotion: only if validation score strictly > champion's validation score
- [ ] Stopping criteria: target accuracy reached, max rounds, plateau, budget exhausted, perfect score
- [ ] Budget tracking: total API cost across all rounds (runner + judge + optimizer), stop if exceeds max_budget_usd
- [ ] Full history persistence: every round saved to disk as it completes
- [ ] Final test evaluation: run champion on frozen test set ONCE at the end (optional, enabled by default)
- [ ] Progress display: show round number, scores, champion status using Rich
- [ ] OptimizationReport with full results, trajectory, cost breakdown
- [ ] Report saved as markdown and JSON
- [ ] Champion config exported as standalone YAML
- [ ] Resume support: can resume a stopped/crashed optimization from the last completed round
- [ ] Graceful KeyboardInterrupt handling: save state, show partial results
- [ ] Async execution (the entire loop is async)

---

## Data Models

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class StoppingReason(str, Enum):
    """Why the optimization loop stopped."""
    TARGET_REACHED = "target_accuracy_reached"
    MAX_ROUNDS = "max_rounds_reached"
    PLATEAU = "accuracy_plateaued"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PERFECT_SCORE = "perfect_score"
    USER_STOPPED = "user_stopped"  # KeyboardInterrupt


class RoundResult(BaseModel):
    """Complete results from a single optimization round.

    Saved to disk after each round completes for resume support.
    """
    round_number: int
    candidate: Candidate
    train_run: RunResult
    train_scores: ScoreCard
    validation_run: RunResult | None = None
    validation_scores: ScoreCard | None = None
    is_champion: bool = False
    proposal: OptimizerProposal | None = None  # None for last round (stopping)
    round_cost_usd: float = 0.0
    round_duration_seconds: float = 0.0


class CostBreakdown(BaseModel):
    """Tracks where money was spent during optimization."""
    runner_train_cost: float = 0.0       # Running candidates on train set
    runner_validation_cost: float = 0.0  # Running candidates on validation set
    runner_test_cost: float = 0.0        # Final test evaluation
    judge_cost: float = 0.0             # LLM grading calls
    optimizer_cost: float = 0.0          # Optimizer proposal calls
    total: float = 0.0


class OptimizationReport(BaseModel):
    """Final report from a completed optimization run.

    Contains everything needed to understand what happened,
    what was found, and how to use the result.
    """
    task_name: str
    run_id: str
    started_at: datetime
    completed_at: datetime
    stopping_reason: StoppingReason
    total_rounds: int
    total_cost: CostBreakdown
    champion: Candidate
    champion_train_scores: ScoreCard
    champion_validation_scores: ScoreCard
    champion_test_scores: ScoreCard | None = None  # None if --no-test
    history: list[RoundResult]

    def to_markdown(self) -> str:
        """Generate a human-readable markdown report."""
        ...  # See Report Format section below

    def to_json(self) -> str:
        """Generate a machine-readable JSON summary."""
        ...

    def accuracy_trajectory(self) -> list[tuple[int, float]]:
        """Return [(round_number, validation_accuracy), ...] for plotting."""
        ...
```

---

## Controller Implementation

```python
import asyncio
import time
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

logger = logging.getLogger(__name__)


class Controller:
    """Orchestrates the full optimization loop.

    This is the main entry point for running an optimization.
    The CLI calls Controller.run() and gets back an OptimizationReport.
    """

    def __init__(
        self,
        task_name: str,
        settings: Settings,
        max_rounds: int = 50,
        target_accuracy: float = 0.95,
        patience: int = 5,
        max_budget_usd: float = 50.0,
        run_test_eval: bool = True,
    ):
        """Initialize the controller with all dependencies.

        Parameters:
            task_name: Name of the task (must exist in datasets/)
            settings: Global settings (API key, paths, etc.)
            max_rounds: Maximum optimization rounds
            target_accuracy: Stop when validation accuracy >= this
            patience: Rounds without improvement before declaring plateau
            max_budget_usd: Maximum total API spend before stopping
            run_test_eval: Whether to run final test set evaluation
        """
        self.task_name = task_name
        self.settings = settings
        self.max_rounds = max_rounds
        self.target_accuracy = target_accuracy
        self.patience = patience
        self.max_budget_usd = max_budget_usd
        self.run_test_eval = run_test_eval

        # Load dataset and config
        self.dataset = Dataset(settings.datasets_dir / task_name)
        if not self.dataset.exists():
            raise FileNotFoundError(f"Task '{task_name}' not found at {self.dataset.task_dir}")
        self.task_config = self.dataset.load_config()

        # Initialize components
        self.provider = AnthropicProvider(api_key=settings.anthropic_api_key)
        self.runner = Runner(self.provider, max_concurrent=settings.max_concurrent_requests)
        self.judge = Judge(self.provider, self.task_config)
        self.optimizer = Optimizer(self.provider, optimizer_model=settings.default_model)
        self.candidate_factory = CandidateFactory(self.task_config)

        # State
        self.history = OptimizationHistory()
        self.round_results: list[RoundResult] = []
        self.total_cost = CostBreakdown()
        self.console = Console()

    async def run(self) -> OptimizationReport:
        """Run the full optimization loop.

        Returns OptimizationReport when complete.
        Handles KeyboardInterrupt gracefully.
        """
        started_at = datetime.utcnow()
        run_id = f"run_{started_at.strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.settings.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save run config
        self._save_run_config(run_dir)

        # Create baseline
        champion = self.candidate_factory.create_baseline()
        champion_val_scores: ScoreCard | None = None
        stopping_reason = StoppingReason.MAX_ROUNDS

        self.console.print(Panel(
            f"Task: {self.task_name}\n"
            f"Max rounds: {self.max_rounds}\n"
            f"Target: {self.target_accuracy:.0%}\n"
            f"Budget: ${self.max_budget_usd:.2f}",
            title="Prompt Optimizer",
        ))

        try:
            for round_num in range(1, self.max_rounds + 1):
                round_start = time.monotonic()
                self.console.rule(f"Round {round_num}/{self.max_rounds}")

                # ── 1. Run champion on TRAIN set ──
                train_examples = self.dataset.load_split("train")
                train_run = await self.runner.run(champion, train_examples, "train", self.dataset)
                train_scores = await self.judge.score(train_run, train_examples)
                self.console.print(f"  Train accuracy:  {train_scores.accuracy:.1%} "
                                   f"(pass rate: {train_scores.pass_rate:.1%})")

                # ── 2. Run champion on VALIDATION set ──
                val_examples = self.dataset.load_split("validation")
                val_run = await self.runner.run(champion, val_examples, "validation", self.dataset)
                champion_val_scores = await self.judge.score(val_run, val_examples)
                self.console.print(f"  Val accuracy:    {champion_val_scores.accuracy:.1%} "
                                   f"(pass rate: {champion_val_scores.pass_rate:.1%})")

                # ── 3. Track cost ──
                round_cost = (
                    train_run.estimated_cost_usd
                    + val_run.estimated_cost_usd
                    # Judge and optimizer costs are harder to track precisely
                    # so we estimate them
                )
                self.total_cost.runner_train_cost += train_run.estimated_cost_usd
                self.total_cost.runner_validation_cost += val_run.estimated_cost_usd
                self.total_cost.total = (
                    self.total_cost.runner_train_cost
                    + self.total_cost.runner_validation_cost
                    + self.total_cost.runner_test_cost
                    + self.total_cost.judge_cost
                    + self.total_cost.optimizer_cost
                )

                self.console.print(f"  Cost this round: ${round_cost:.4f} "
                                   f"(total: ${self.total_cost.total:.4f})")

                # ── 4. Record round in history ──
                round_summary = RoundSummary(
                    round_number=round_num,
                    candidate_id=champion.id,
                    model=champion.model,
                    temperature=champion.temperature,
                    system_prompt_preview=champion.system_prompt[:200],
                    train_accuracy=train_scores.accuracy,
                    train_pass_rate=train_scores.pass_rate,
                    validation_accuracy=champion_val_scores.accuracy,
                    validation_pass_rate=champion_val_scores.pass_rate,
                    is_champion=True,
                    changes_from_previous=champion.reasoning[:100] if round_num > 1 else "baseline",
                )
                self.history.add_round(round_summary)

                # ── 5. Check stopping criteria ──
                if champion_val_scores.accuracy >= self.target_accuracy:
                    stopping_reason = StoppingReason.TARGET_REACHED
                    self.console.print(f"\n[bold green]Target accuracy {self.target_accuracy:.0%} reached![/bold green]")
                    round_result = RoundResult(
                        round_number=round_num, candidate=champion,
                        train_run=train_run, train_scores=train_scores,
                        validation_run=val_run, validation_scores=champion_val_scores,
                        is_champion=True, round_cost_usd=round_cost,
                        round_duration_seconds=time.monotonic() - round_start,
                    )
                    self.round_results.append(round_result)
                    self._save_round(run_dir, round_result)
                    break

                if champion_val_scores.accuracy >= 1.0:
                    stopping_reason = StoppingReason.PERFECT_SCORE
                    self.console.print("\n[bold green]Perfect score![/bold green]")
                    round_result = RoundResult(
                        round_number=round_num, candidate=champion,
                        train_run=train_run, train_scores=train_scores,
                        validation_run=val_run, validation_scores=champion_val_scores,
                        is_champion=True, round_cost_usd=round_cost,
                        round_duration_seconds=time.monotonic() - round_start,
                    )
                    self.round_results.append(round_result)
                    self._save_round(run_dir, round_result)
                    break

                if self.total_cost.total >= self.max_budget_usd:
                    stopping_reason = StoppingReason.BUDGET_EXHAUSTED
                    self.console.print(f"\n[bold yellow]Budget exhausted (${self.max_budget_usd:.2f})[/bold yellow]")
                    round_result = RoundResult(
                        round_number=round_num, candidate=champion,
                        train_run=train_run, train_scores=train_scores,
                        validation_run=val_run, validation_scores=champion_val_scores,
                        is_champion=True, round_cost_usd=round_cost,
                        round_duration_seconds=time.monotonic() - round_start,
                    )
                    self.round_results.append(round_result)
                    self._save_round(run_dir, round_result)
                    break

                if self.history.is_plateaued(patience=self.patience):
                    stopping_reason = StoppingReason.PLATEAU
                    self.console.print(f"\n[bold yellow]Accuracy plateaued for {self.patience} rounds[/bold yellow]")
                    round_result = RoundResult(
                        round_number=round_num, candidate=champion,
                        train_run=train_run, train_scores=train_scores,
                        validation_run=val_run, validation_scores=champion_val_scores,
                        is_champion=True, round_cost_usd=round_cost,
                        round_duration_seconds=time.monotonic() - round_start,
                    )
                    self.round_results.append(round_result)
                    self._save_round(run_dir, round_result)
                    break

                # ── 6. Optimizer proposes challenger ──
                self.console.print("  Optimizer analyzing...")
                proposal = await self.optimizer.propose(
                    current=champion,
                    scorecard=train_scores,
                    history=self.history,
                    task_config=self.task_config,
                    dataset=self.dataset,
                )
                self.console.print(f"  Proposal: {proposal.changes_summary}")

                challenger = self.candidate_factory.create_from_proposal(proposal, champion)

                # ── 7. Run challenger on VALIDATION set ──
                self.console.print(f"  Testing challenger ({challenger.id})...")
                challenger_val_run = await self.runner.run(challenger, val_examples, "validation", self.dataset)
                challenger_val_scores = await self.judge.score(challenger_val_run, val_examples)
                self.total_cost.runner_validation_cost += challenger_val_run.estimated_cost_usd
                self.total_cost.total += challenger_val_run.estimated_cost_usd

                # ── 8. Promote if better ──
                if challenger_val_scores.accuracy > champion_val_scores.accuracy:
                    self.console.print(
                        f"  [bold green]New champion! "
                        f"{champion_val_scores.accuracy:.1%} → {challenger_val_scores.accuracy:.1%}[/bold green]"
                    )
                    champion = challenger
                    champion_val_scores = challenger_val_scores
                else:
                    self.console.print(
                        f"  Champion holds. "
                        f"({champion_val_scores.accuracy:.1%} vs challenger {challenger_val_scores.accuracy:.1%})"
                    )

                # ── 9. Save round ──
                round_result = RoundResult(
                    round_number=round_num,
                    candidate=champion,
                    train_run=train_run,
                    train_scores=train_scores,
                    validation_run=val_run,
                    validation_scores=champion_val_scores,
                    is_champion=True,
                    proposal=proposal,
                    round_cost_usd=round_cost,
                    round_duration_seconds=time.monotonic() - round_start,
                )
                self.round_results.append(round_result)
                self._save_round(run_dir, round_result)

                # Overfitting warning
                if train_scores.accuracy - champion_val_scores.accuracy > 0.10:
                    self.console.print(
                        f"  [yellow]⚠ Possible overfitting: train {train_scores.accuracy:.1%} "
                        f"vs val {champion_val_scores.accuracy:.1%}[/yellow]"
                    )

        except KeyboardInterrupt:
            stopping_reason = StoppingReason.USER_STOPPED
            self.console.print("\n[yellow]Optimization stopped by user.[/yellow]")

        # ── Final test evaluation ──
        test_scores = None
        if self.run_test_eval and champion_val_scores is not None:
            self.console.print("\n[bold]Running final test evaluation...[/bold]")
            test_examples = self.dataset.load_split("test")
            test_run = await self.runner.run(champion, test_examples, "test", self.dataset)
            test_scores = await self.judge.score(test_run, test_examples)
            self.total_cost.runner_test_cost += test_run.estimated_cost_usd
            self.total_cost.total += test_run.estimated_cost_usd
            self.console.print(f"[bold]Test accuracy: {test_scores.accuracy:.1%}[/bold]")

        completed_at = datetime.utcnow()

        # Build report
        report = OptimizationReport(
            task_name=self.task_name,
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            stopping_reason=stopping_reason,
            total_rounds=len(self.round_results),
            total_cost=self.total_cost,
            champion=champion,
            champion_train_scores=self.round_results[-1].train_scores if self.round_results else train_scores,
            champion_validation_scores=champion_val_scores,
            champion_test_scores=test_scores,
            history=self.round_results,
        )

        # Save report and champion
        self._save_report(run_dir, report)
        self._save_champion(run_dir, champion)

        # Print summary
        self._print_summary(report)

        # Cleanup
        await self.provider.close()

        return report

    def _save_run_config(self, run_dir: Path) -> None:
        """Save the run configuration as YAML."""
        ...

    def _save_round(self, run_dir: Path, round_result: RoundResult) -> None:
        """Save round data to disk.

        Creates: runs/<run_id>/rounds/round_NNN/
          - candidate.yaml
          - train_scores.json
          - validation_scores.json
          - proposal.json (if any)
        """
        round_dir = run_dir / "rounds" / f"round_{round_result.round_number:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        # Save candidate, scores, proposal
        ...

    def _save_report(self, run_dir: Path, report: OptimizationReport) -> None:
        """Save the final report as markdown and JSON."""
        (run_dir / "report.md").write_text(report.to_markdown())
        (run_dir / "summary.json").write_text(report.to_json())

    def _save_champion(self, run_dir: Path, champion: Candidate) -> None:
        """Save the champion config as standalone YAML."""
        save_candidate(champion, run_dir / "champion.yaml")

    def _print_summary(self, report: OptimizationReport) -> None:
        """Print a final summary table to the terminal."""
        ...
```

---

## Run Directory Structure

```
runs/
  run_20260322_143000/
    config.yaml                      # Snapshot of run parameters
    rounds/
      round_001/
        candidate.yaml               # The candidate tested this round
        train_results.jsonl           # Raw runner outputs on train
        train_scores.json             # Judge scorecard for train
        validation_results.jsonl      # Raw runner outputs on validation
        validation_scores.json        # Judge scorecard for validation
        proposal.json                 # Optimizer's proposal (if any)
      round_002/
        ...
    champion.yaml                    # Final champion config (standalone)
    report.md                        # Human-readable report
    summary.json                     # Machine-readable summary
```

---

## Stopping Criteria

| Criterion | Condition | StoppingReason |
|-----------|-----------|---------------|
| Target reached | `champion_val_scores.accuracy >= target_accuracy` | `TARGET_REACHED` |
| Perfect score | `champion_val_scores.accuracy >= 1.0` | `PERFECT_SCORE` |
| Max rounds | `round_num >= max_rounds` | `MAX_ROUNDS` |
| Plateau | Validation accuracy variance < 1% over last `patience` rounds | `PLATEAU` |
| Budget exhausted | `total_cost.total >= max_budget_usd` | `BUDGET_EXHAUSTED` |
| User stopped | `KeyboardInterrupt` caught | `USER_STOPPED` |

**Overfitting detection** (warning only, does not stop):
If `train_accuracy - validation_accuracy > 0.10`, print a warning.

---

## Report Format (Markdown)

```markdown
# Optimization Report: {task_name}

## Summary
| Metric | Value |
|--------|-------|
| Run ID | run_20260322_143000 |
| Rounds | 23 |
| Duration | 45 minutes |
| Total cost | $3.47 |
| Stopping reason | Target accuracy reached |

## Champion Configuration
- **Model:** claude-sonnet-4-20250514
- **Temperature:** 0.0
- **Max tokens:** 50
- **Train accuracy:** 94.2%
- **Validation accuracy:** 95.1%
- **Test accuracy:** 93.8%

### System Prompt
```
{champion_system_prompt}
```

## Optimization Trajectory
| Round | Val Accuracy | Model | Change Summary | Champion? |
|-------|-------------|-------|----------------|-----------|
| 1 | 65.0% | sonnet | Baseline | Yes |
| 2 | 72.0% | sonnet | Added format constraint | Yes |
| 3 | 72.5% | sonnet | Added explicit category list | Yes |
| ... | ... | ... | ... | ... |

## Cost Breakdown
| Component | Cost |
|-----------|------|
| Runner (train) | $1.20 |
| Runner (validation) | $0.85 |
| Runner (test) | $0.12 |
| Judge (LLM grading) | $0.90 |
| Optimizer | $0.40 |
| **Total** | **$3.47** |

## Final Test Results
- Accuracy: 93.8%
- Pass rate: 92.0%
- Failed: 2/30

### Test Failures
| Example ID | Expected | Actual | Reason |
|-----------|----------|--------|--------|
| ex_142 | shipping | shipping_delay | Near-miss: related category |
| ex_187 | other | account | Ambiguous input |
```

---

## Resume Support

### Save state after each round
After each round completes, the round data is written to disk immediately. This means:
- If the process crashes, all completed rounds are preserved
- On resume, load the run directory, find the last completed round, continue from round N+1

### Resume flow
```python
@classmethod
async def resume(cls, run_id: str, settings: Settings) -> "Controller":
    """Resume a previously stopped optimization run.

    1. Load run config from runs/<run_id>/config.yaml
    2. Load all completed round results
    3. Reconstruct history and champion
    4. Continue from the next round
    """
    ...
```

---

## Progress Display

During optimization, show a Rich panel that updates each round:

```
╭─── Prompt Optimizer ─── Round 5/50 ───────────────────╮
│                                                       │
│ Champion: candidate_003                               │
│ Model:    claude-sonnet-4-20250514                    │
│ Val accuracy: 82.0%                                   │
│ Budget:   $1.23 / $50.00                              │
│                                                       │
│ Status: Running validation for challenger...          │
│                                                       │
╰───────────────────────────────────────────────────────╯
```

---

## Testing Requirements

```
# Full loop (all components mocked)
test_run_full_loop_reaches_target
test_run_full_loop_max_rounds
test_run_full_loop_plateau_stops
test_run_full_loop_budget_stops
test_run_full_loop_perfect_score_stops

# Champion tracking
test_champion_promoted_when_better
test_champion_holds_when_worse
test_champion_holds_on_tie

# Stopping criteria
test_stopping_target_reached
test_stopping_max_rounds
test_stopping_plateau_detection
test_stopping_budget_exhausted
test_stopping_keyboard_interrupt

# Cost tracking
test_cost_tracking_accumulates
test_cost_breakdown_correct

# History
test_history_records_each_round
test_history_marks_champion

# Persistence
test_save_round_creates_files
test_save_report_markdown
test_save_champion_yaml
test_resume_loads_state

# Report
test_report_to_markdown
test_report_accuracy_trajectory
test_report_includes_all_rounds

# Edge cases
test_run_with_no_validation_split  # should error gracefully
test_run_with_no_train_split       # should error gracefully
test_overfitting_warning_logged
test_run_with_test_eval_disabled
```

Mock Runner, Judge, and Optimizer. Test the Controller's orchestration logic in isolation.

**Minimum: 25 test cases.**

---

## Implementation Notes

- The entire Controller.run() is async — it awaits Runner, Judge, and Optimizer calls
- Use `try/except KeyboardInterrupt` around the main loop to handle graceful shutdown
- Save state after EVERY completed round (not batched) — this enables resume
- The Controller does NOT make any LLM calls directly — it delegates to Runner, Judge, and Optimizer
- Validation set is run for BOTH champion (to track progress) AND challenger (to compare)
- The first round is special: the champion IS the baseline, there is no challenger yet
- Cost tracking is approximate — exact LLM costs depend on caching and pricing changes

---

## Dependencies

- **Depends on:** ALL previous tickets (1-6)
- **Depended on by:** Ticket 8 (CLI calls Controller.run())
