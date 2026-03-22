# Ticket 6: Optimizer (Proposal Engine)

## Overview

The Optimizer is the "brain" of the system — an LLM that analyzes evaluation results from the Judge and proposes better candidate configurations. It reads failure patterns, reviews the full optimization history, and suggests changes to the system prompt, model, temperature, or few-shot examples.

It's the component that makes the loop intelligent rather than random search.

---

## Acceptance Criteria

- [ ] `Optimizer.propose(current, scorecard, history, task_config, dataset) -> OptimizerProposal`
- [ ] The optimizer prompt includes: current config, failure clusters, sample failures, history summary, constraints
- [ ] Output is a structured `OptimizerProposal` with new config + reasoning
- [ ] **The optimizer CANNOT modify:** scoring function, expected outputs, judge config, test set access
- [ ] Staged search strategy: try prompt changes first, switch models only after prompt search plateaus
- [ ] History summarization: condense history to fit in context window (don't send 50 rounds verbatim)
- [ ] Constraint enforcement: proposed model must be in allowlist, temperature in range, prompt under length limit
- [ ] If the optimizer returns invalid JSON or invalid config, retry once with the error message
- [ ] If retry also fails, fall back to a simple mutation (minor rephrase of current prompt)
- [ ] Diversity injection: if stuck at same score for 3+ rounds, instruct optimizer to try a radically different approach
- [ ] The optimizer model itself is configurable in Settings (defaults to claude-sonnet-4-20250514)

---

## Data Models

```python
from pydantic import BaseModel, Field


class OptimizerProposal(BaseModel):
    """The optimizer's proposed next candidate configuration.

    This is the output of the Optimizer — the Controller uses it
    to create a new Candidate via CandidateFactory.
    """
    model: str                           # Proposed model ID
    system_prompt: str                   # Proposed system prompt (full text)
    temperature: float                   # Proposed temperature
    max_tokens: int                      # Proposed max tokens
    few_shot_example_ids: list[str] = [] # Proposed few-shot example IDs
    reasoning: str                       # Why these changes should help
    changes_summary: str                 # 1-line summary of what changed
    confidence: float = 0.5             # 0.0-1.0 how confident the optimizer is


class RoundSummary(BaseModel):
    """Summary of a single optimization round for history tracking."""
    round_number: int
    candidate_id: str
    model: str
    temperature: float
    system_prompt_preview: str           # First 200 chars of system prompt
    train_accuracy: float
    train_pass_rate: float
    validation_accuracy: float | None = None
    validation_pass_rate: float | None = None
    is_champion: bool = False
    changes_from_previous: str = ""      # What was changed this round


class OptimizationHistory(BaseModel):
    """Complete history of all optimization rounds.

    Used to give the optimizer context about what's been tried
    and how well each attempt performed.
    """
    rounds: list[RoundSummary] = []

    def add_round(self, summary: RoundSummary) -> None:
        self.rounds.append(summary)

    @property
    def best_accuracy(self) -> float:
        if not self.rounds:
            return 0.0
        val_scores = [r.validation_accuracy for r in self.rounds if r.validation_accuracy is not None]
        return max(val_scores) if val_scores else max(r.train_accuracy for r in self.rounds)

    @property
    def best_round(self) -> RoundSummary | None:
        if not self.rounds:
            return None
        champion_rounds = [r for r in self.rounds if r.is_champion]
        return champion_rounds[-1] if champion_rounds else self.rounds[0]

    def is_plateaued(self, patience: int = 3) -> bool:
        """Check if accuracy has been flat for `patience` consecutive rounds."""
        if len(self.rounds) < patience:
            return False
        recent = self.rounds[-patience:]
        scores = [r.validation_accuracy or r.train_accuracy for r in recent]
        return max(scores) - min(scores) < 0.01

    def summarize(self, max_recent: int = 10) -> str:
        """Produce a text summary for the optimizer prompt.

        Strategy:
        - If <= max_recent rounds: include all with full details
        - If > max_recent rounds: summarize early rounds, detail recent ones
        """
        if not self.rounds:
            return "No previous rounds."

        lines = []

        if len(self.rounds) <= max_recent:
            # Include all rounds
            for r in self.rounds:
                champion_marker = " ← CHAMPION" if r.is_champion else ""
                val_str = f", val: {r.validation_accuracy:.1%}" if r.validation_accuracy is not None else ""
                lines.append(
                    f"Round {r.round_number}: train: {r.train_accuracy:.1%}{val_str} "
                    f"| {r.model} | {r.changes_from_previous or 'baseline'}{champion_marker}"
                )
        else:
            # Summarize early rounds
            early = self.rounds[:-max_recent]
            early_best = max(r.validation_accuracy or r.train_accuracy for r in early)
            early_worst = min(r.validation_accuracy or r.train_accuracy for r in early)
            lines.append(
                f"Rounds 1-{len(early)}: accuracy ranged from {early_worst:.1%} to {early_best:.1%}"
            )
            lines.append("")

            # Detail recent rounds
            for r in self.rounds[-max_recent:]:
                champion_marker = " ← CHAMPION" if r.is_champion else ""
                val_str = f", val: {r.validation_accuracy:.1%}" if r.validation_accuracy is not None else ""
                lines.append(
                    f"Round {r.round_number}: train: {r.train_accuracy:.1%}{val_str} "
                    f"| {r.model} | {r.changes_from_previous or 'baseline'}{champion_marker}"
                )

        # Always include best
        best = self.best_round
        if best:
            lines.append(f"\nBest so far: Round {best.round_number} "
                         f"({best.validation_accuracy or best.train_accuracy:.1%})")

        return "\n".join(lines)
```

---

## The Optimizer Prompt

This is the most critical piece — the exact prompt template that guides the optimizer's reasoning.

```python
OPTIMIZER_SYSTEM_PROMPT = "You are an expert LLM prompt engineer and optimizer. Your job is to analyze evaluation results and propose improved configurations."

OPTIMIZER_PROMPT_TEMPLATE = '''Analyze the evaluation results below and propose an improved candidate configuration.

## CURRENT CANDIDATE (Round {round_number})
Model: {model}
Temperature: {temperature}
Max tokens: {max_tokens}
System prompt:
```
{system_prompt}
```
Few-shot examples: {few_shot_count} examples
{few_shot_details}

## EVALUATION RESULTS
Split: {split}
Overall accuracy: {accuracy:.1%}
Pass rate: {pass_rate:.1%}
Total examples: {total_examples}
Passed: {passed} | Failed: {failed}

## FAILURE ANALYSIS
{failure_clusters}

## SAMPLE FAILURES (showing up to 10)
{sample_failures}

## OPTIMIZATION HISTORY
{history_summary}

## CONSTRAINTS
- Allowed models: {allowed_models}
- Temperature range: 0.0 - 1.0
- Max system prompt length: 10000 characters
- Max few-shot examples: 5
- Available few-shot example IDs: {available_few_shot_ids}

## STRATEGY GUIDELINES
1. Focus on PROMPT CHANGES first. Only switch models if prompt changes have plateaued for 3+ rounds.
2. Address the LARGEST failure cluster first — fixing the most common error pattern gives the biggest accuracy boost.
3. Be SPECIFIC in the system prompt. Vague instructions lead to inconsistent outputs.
4. If the model is adding unwanted text (explanations, caveats), add explicit constraints: "respond with ONLY X, nothing else."
5. If the model is misformatting output, add an explicit format specification or example in the system prompt.
6. Consider adding few-shot examples to demonstrate desired behavior, especially for edge cases.
7. If stuck at the same accuracy for 3+ rounds, try a fundamentally different prompting approach (different framing, structure, or constraints).
8. Lower temperature (closer to 0.0) for deterministic/classification tasks, higher for creative/generation tasks.
9. Review what previous rounds changed — don't repeat a strategy that already failed.

{diversity_injection}

## YOUR RESPONSE
Respond with a JSON object in this EXACT format (no other text before or after):
{{
  "reasoning": "Your analysis of the failures and why your proposed changes should fix them",
  "changes_summary": "One-line summary of what you changed",
  "confidence": <float 0.0-1.0>,
  "proposed_config": {{
    "model": "{default_model}",
    "system_prompt": "The complete new system prompt text",
    "temperature": 0.0,
    "max_tokens": 1024,
    "few_shot_example_ids": []
  }}
}}'''
```

---

## Optimizer Implementation

```python
import json
import logging

logger = logging.getLogger(__name__)


class OptimizerError(Exception):
    """Raised when the optimizer fails to produce a valid proposal."""
    pass


class Optimizer:
    """Analyzes evaluation results and proposes improved candidate configurations.

    The optimizer:
    1. Receives the current candidate's scorecard and failure clusters
    2. Reviews the optimization history
    3. Proposes a new candidate config (prompt, model, temperature, few-shot)

    The optimizer CANNOT modify scoring criteria, expected outputs, or the judge.
    """

    def __init__(
        self,
        provider: AnthropicProvider,
        optimizer_model: str = "claude-sonnet-4-20250514",
    ):
        self.provider = provider
        self.optimizer_model = optimizer_model

    async def propose(
        self,
        current: Candidate,
        scorecard: ScoreCard,
        history: OptimizationHistory,
        task_config: TaskConfig,
        dataset: Dataset,
    ) -> OptimizerProposal:
        """Analyze results and propose a new candidate configuration.

        Process:
        1. Build the optimizer prompt with all context
        2. Call the optimizer LLM
        3. Parse the JSON response
        4. Validate the proposal against task constraints
        5. Return the proposal

        If parsing fails, retry once with the error message.
        If retry also fails, fall back to a simple prompt mutation.
        """
        round_number = len(history.rounds) + 1
        prompt = self._build_prompt(current, scorecard, history, task_config, dataset, round_number)

        # Determine optimizer temperature
        optimizer_temp = 0.7
        if history.is_plateaued(patience=3):
            optimizer_temp = 0.9  # more creative when stuck

        # First attempt
        try:
            response = await self.provider.complete(
                model=self.optimizer_model,
                system_prompt=OPTIMIZER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=optimizer_temp,
                max_tokens=4096,
            )
            proposal = self._parse_response(response.content)
            self._validate_proposal(proposal, task_config)
            return proposal

        except (json.JSONDecodeError, KeyError, ValueError, OptimizerError) as e:
            logger.warning(f"Optimizer first attempt failed: {e}. Retrying...")

            # Retry with error feedback
            try:
                retry_response = await self.provider.complete(
                    model=self.optimizer_model,
                    system_prompt=OPTIMIZER_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response.content if 'response' in dir() else ""},
                        {"role": "user", "content": f"Your response could not be parsed. Error: {e}\n\nPlease respond with ONLY a valid JSON object in the specified format. No markdown, no explanation outside the JSON."},
                    ],
                    temperature=0.3,  # lower temp for retry
                    max_tokens=4096,
                )
                proposal = self._parse_response(retry_response.content)
                self._validate_proposal(proposal, task_config)
                return proposal

            except Exception as retry_error:
                logger.warning(f"Optimizer retry also failed: {retry_error}. Using fallback mutation.")
                return self._fallback_mutation(current, task_config)

    def _build_prompt(
        self,
        current: Candidate,
        scorecard: ScoreCard,
        history: OptimizationHistory,
        task_config: TaskConfig,
        dataset: Dataset,
        round_number: int,
    ) -> str:
        """Construct the full optimizer prompt with all context."""
        # Few-shot details
        if current.few_shot_example_ids:
            few_shot_details = "Included examples:\n"
            for eid in current.few_shot_example_ids:
                ex = dataset.get_example_by_id(eid)
                if ex:
                    few_shot_details += f"  - {eid}: input='{ex.input[:100]}...' output='{ex.expected_output}'\n"
        else:
            few_shot_details = "(none)"

        # Available few-shot IDs (from train split only)
        try:
            train_examples = dataset.load_split("train")
            available_ids = [e.id for e in train_examples[:50]]  # cap at 50
            available_ids_str = ", ".join(available_ids)
        except Exception:
            available_ids_str = "(unable to load)"

        # Diversity injection
        diversity_injection = ""
        if history.is_plateaued(patience=3):
            diversity_injection = (
                "## IMPORTANT: PLATEAU DETECTED\n"
                "Your last 3+ proposals have not improved accuracy. "
                "You MUST try a fundamentally different approach:\n"
                "- Completely restructure the system prompt (different framing)\n"
                "- Try a different model from the allowed list\n"
                "- Add or remove few-shot examples\n"
                "- Change the output format instruction\n"
                "- Try a higher temperature for creative tasks or lower for deterministic\n"
                "DO NOT make minor tweaks — make a bold change.\n"
            )

        return OPTIMIZER_PROMPT_TEMPLATE.format(
            round_number=round_number,
            model=current.model,
            temperature=current.temperature,
            max_tokens=current.max_tokens,
            system_prompt=current.system_prompt,
            few_shot_count=len(current.few_shot_example_ids),
            few_shot_details=few_shot_details,
            split=scorecard.split,
            accuracy=scorecard.accuracy,
            pass_rate=scorecard.pass_rate,
            total_examples=scorecard.total_examples,
            passed=scorecard.total_passed,
            failed=scorecard.total_failed,
            failure_clusters=self._format_failure_clusters(scorecard.failure_clusters),
            sample_failures=self._format_sample_failures(scorecard),
            history_summary=history.summarize(max_recent=10),
            allowed_models=", ".join(task_config.allowed_models),
            available_few_shot_ids=available_ids_str,
            diversity_injection=diversity_injection,
            default_model=current.model,
        )

    def _format_failure_clusters(self, clusters: list[FailureCluster]) -> str:
        """Format failure clusters for the optimizer prompt."""
        if not clusters:
            return "(No failure clusters identified)"

        lines = []
        for i, c in enumerate(clusters, 1):
            lines.append(f"### Cluster {i}: {c.pattern} ({c.count} examples)")
            lines.append(f"   Example IDs: {', '.join(c.example_ids[:10])}")
            if c.suggested_fix:
                lines.append(f"   Suggested fix: {c.suggested_fix}")
            lines.append("")
        return "\n".join(lines)

    def _format_sample_failures(self, scorecard: ScoreCard, max_samples: int = 10) -> str:
        """Format sample failures for the optimizer prompt."""
        failed = scorecard.failed_scores[:max_samples]
        if not failed:
            return "(No failures)"

        lines = []
        for s in failed:
            lines.append(f"- {s.example_id}: {s.failure_reason}")
        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> OptimizerProposal:
        """Parse the optimizer's JSON response into an OptimizerProposal.

        Handles:
        - Raw JSON
        - JSON wrapped in markdown code blocks (```json ... ```)
        - JSON with surrounding text
        """
        text = response_text.strip()

        # Try to extract JSON from code blocks
        if "```" in text:
            # Find JSON between code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1).strip()

        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        data = json.loads(text)

        config = data["proposed_config"]
        return OptimizerProposal(
            model=config["model"],
            system_prompt=config["system_prompt"],
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 1024)),
            few_shot_example_ids=config.get("few_shot_example_ids", []),
            reasoning=data.get("reasoning", ""),
            changes_summary=data.get("changes_summary", ""),
            confidence=float(data.get("confidence", 0.5)),
        )

    def _validate_proposal(self, proposal: OptimizerProposal, task_config: TaskConfig) -> None:
        """Validate the proposal against task constraints.

        Raises OptimizerError if invalid.
        """
        if proposal.model not in task_config.allowed_models:
            raise OptimizerError(
                f"Proposed model '{proposal.model}' not in allowed models: "
                f"{task_config.allowed_models}"
            )
        if not 0.0 <= proposal.temperature <= 1.0:
            raise OptimizerError(f"Temperature {proposal.temperature} out of range [0.0, 1.0]")
        if not 1 <= proposal.max_tokens <= 8192:
            raise OptimizerError(f"Max tokens {proposal.max_tokens} out of range [1, 8192]")
        if not proposal.system_prompt.strip():
            raise OptimizerError("System prompt must not be empty")
        if len(proposal.system_prompt) > 10000:
            raise OptimizerError(f"System prompt too long: {len(proposal.system_prompt)} > 10000 chars")
        if len(proposal.few_shot_example_ids) > 5:
            raise OptimizerError(f"Too many few-shot examples: {len(proposal.few_shot_example_ids)} > 5")

    def _fallback_mutation(self, current: Candidate, task_config: TaskConfig) -> OptimizerProposal:
        """Generate a simple mutation when the optimizer LLM fails.

        Makes a minor change to the system prompt (adds a clarification line)
        so the loop can continue even if the optimizer is having trouble.
        """
        return OptimizerProposal(
            model=current.model,
            system_prompt=current.system_prompt + "\n\nBe precise and follow the instructions exactly.",
            temperature=current.temperature,
            max_tokens=current.max_tokens,
            few_shot_example_ids=list(current.few_shot_example_ids),
            reasoning="Fallback mutation: optimizer failed to produce a valid proposal. Adding generic precision instruction.",
            changes_summary="Fallback: added generic precision instruction",
            confidence=0.1,
        )
```

---

## History Summarization Strategy

| History Length | Strategy |
|--------------|----------|
| 1-10 rounds | Include all rounds with full details |
| 11-20 rounds | Include all but truncate system prompts to first 200 chars |
| 20+ rounds | Summarize rounds 1-N as a range, detail last 10 rounds |

Always include:
- Best accuracy achieved and which round
- Current champion details
- The last 3 rounds in full detail (most important for the optimizer)

---

## Diversity Injection

**Trigger:** `OptimizationHistory.is_plateaued(patience=3)` returns True (validation accuracy variance < 1% over last 3 rounds).

**Action:**
1. Add a `## IMPORTANT: PLATEAU DETECTED` section to the optimizer prompt
2. Increase the optimizer's own temperature from 0.7 to 0.9
3. Instruct it to make bold changes, not minor tweaks

**Purpose:** Prevent the optimizer from getting stuck in local optima by making the same type of small changes repeatedly.

---

## Error Recovery Flow

```
Attempt 1: Build prompt → Call LLM → Parse JSON → Validate
  │
  ├── Success → return proposal
  │
  └── Failure (parse error, validation error)
      │
      Attempt 2: Send error message → Call LLM again → Parse → Validate
        │
        ├── Success → return proposal
        │
        └── Failure
            │
            Fallback: return simple mutation of current candidate
            (append generic instruction to system prompt)
```

This ensures the optimization loop NEVER crashes due to optimizer failures. It always has a candidate to try, even if it's a weak one.

---

## Testing Requirements

```
# Prompt construction
test_build_prompt_includes_current_config
test_build_prompt_includes_failure_clusters
test_build_prompt_includes_history
test_build_prompt_includes_constraints
test_build_prompt_diversity_injection_when_plateaued
test_build_prompt_no_diversity_when_not_plateaued
test_build_prompt_few_shot_details

# Response parsing
test_parse_response_valid_json
test_parse_response_json_in_code_block
test_parse_response_json_with_surrounding_text
test_parse_response_invalid_json_raises
test_parse_response_missing_fields_raises

# Validation
test_validate_valid_proposal
test_validate_invalid_model_raises
test_validate_temperature_out_of_range_raises
test_validate_empty_prompt_raises
test_validate_prompt_too_long_raises
test_validate_too_many_few_shot_raises

# History summarization
test_history_summarize_empty
test_history_summarize_few_rounds
test_history_summarize_many_rounds_condenses
test_history_best_accuracy
test_history_is_plateaued_true
test_history_is_plateaued_false
test_history_is_plateaued_not_enough_rounds

# Full propose flow
test_propose_success (mock LLM)
test_propose_retry_on_parse_error (mock two calls)
test_propose_fallback_on_total_failure (mock failing calls)
test_propose_higher_temp_when_plateaued

# Fallback mutation
test_fallback_mutation_preserves_model
test_fallback_mutation_appends_to_prompt
test_fallback_mutation_low_confidence
```

Mock ALL LLM calls. Use `pytest.mark.asyncio`.

**Minimum: 25 test cases.**

---

## Dependencies

- **Depends on:** Ticket 1 (config, Settings), Ticket 2 (TaskConfig, Dataset), Ticket 3 (Candidate), Ticket 4 (AnthropicProvider), Ticket 5 (ScoreCard, FailureCluster)
- **Depended on by:** Ticket 7 (Controller calls Optimizer each round)
