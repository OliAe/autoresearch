# Ticket 5: Judge (Scoring Engine)

## Overview

The Judge scores each output from the Runner against the expected output. It determines if the model's response is correct. Two scoring layers:

1. **Hard checks** (deterministic, fast, trustworthy) — exact match, contains, regex, etc.
2. **LLM grader** (for fuzzy/subjective evaluation) — uses a separate LLM call with a fixed grading prompt

The Judge's scoring criteria are **HUMAN-DEFINED and LOCKED** — the Optimizer can never modify them. This is the critical firewall that prevents reward hacking.

---

## Acceptance Criteria

- [ ] `Judge.score(run_result, examples) -> ScoreCard` scores all outputs
- [ ] Hard check framework: exact_match, case-insensitive match, must_contain, must_not_contain, regex_match
- [ ] Custom check support: user provides a dotted Python path to a function that returns `bool`
- [ ] LLM grader: uses a FIXED grading prompt to score fuzzy outputs on a 0.0-1.0 scale
- [ ] LLM grader uses a **different model** than the candidate being evaluated (prevent self-grading bias)
- [ ] Grading prompt is defined in `TaskConfig.scoring` and is IMMUTABLE during optimization
- [ ] Composite scoring: `score = hard_score * hard_weight + soft_score * soft_weight`
- [ ] Weights come from `TaskConfig.scoring` (default: 0.7 hard, 0.3 soft)
- [ ] Per-example output: score (0.0-1.0), pass/fail, explanation of failure
- [ ] Aggregate output: accuracy, pass_rate, total_passed, total_failed
- [ ] Failure clustering: group failed examples by similar error patterns (LLM-assisted)
- [ ] ScoreCard saved to disk as JSON
- [ ] If no hard checks are configured on an example, hard check score defaults to 1.0 (auto-pass)
- [ ] If no LLM grading is configured (weight = 0), skip LLM grading entirely

---

## Data Models

```python
from pydantic import BaseModel, Field
from datetime import datetime


class ExampleScore(BaseModel):
    """Score for a single example output.

    Contains both the hard check results and optional LLM grade,
    combined into a composite score.
    """
    example_id: str
    hard_check_score: float                  # 0.0-1.0 (average of all hard checks)
    hard_check_details: dict[str, bool]      # {"exact_match": True, "contains_billing": False}
    llm_grade: float | None = None           # 0.0-1.0, None if LLM grading not used
    llm_grade_explanation: str = ""           # LLM grader's explanation
    composite_score: float                   # Weighted combination of hard + soft
    passed: bool                             # composite_score >= pass_threshold
    failure_reason: str = ""                 # Human-readable explanation of why it failed


class FailureCluster(BaseModel):
    """A group of failures sharing a common pattern.

    The Optimizer uses these clusters to diagnose systematic issues
    and propose targeted fixes.
    """
    pattern: str                             # e.g., "Model adds explanation after the answer"
    example_ids: list[str]                   # Which examples hit this pattern
    count: int                               # Number of examples in this cluster
    representative_example: str = ""         # One example showing the pattern
    suggested_fix: str = ""                  # Hint for the optimizer


class ScoreCard(BaseModel):
    """Complete scoring results for a candidate on a split.

    This is what the Optimizer receives to analyze performance.
    """
    candidate_id: str
    run_id: str
    split: str                               # "train", "validation", or "test"
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    example_scores: list[ExampleScore]
    failure_clusters: list[FailureCluster] = []

    @property
    def accuracy(self) -> float:
        """Mean composite score across all examples."""
        if not self.example_scores:
            return 0.0
        return sum(s.composite_score for s in self.example_scores) / len(self.example_scores)

    @property
    def pass_rate(self) -> float:
        """Fraction of examples that passed (score >= threshold)."""
        if not self.example_scores:
            return 0.0
        return self.total_passed / len(self.example_scores)

    @property
    def total_passed(self) -> int:
        return sum(1 for s in self.example_scores if s.passed)

    @property
    def total_failed(self) -> int:
        return sum(1 for s in self.example_scores if not s.passed)

    @property
    def total_examples(self) -> int:
        return len(self.example_scores)

    @property
    def failed_scores(self) -> list[ExampleScore]:
        """All scores that did not pass."""
        return [s for s in self.example_scores if not s.passed]
```

---

## Hard Check Implementation

```python
import re
import importlib


class HardChecker:
    """Runs deterministic checks on model outputs.

    Hard checks are fast, reliable, and unambiguous. They should be
    preferred over LLM grading wherever the task allows.
    """

    @staticmethod
    def check(actual: str, expected: str, config: CheckConfig) -> dict[str, bool]:
        """Run all configured hard checks and return results.

        Parameters:
            actual: The model's actual output
            expected: The expected output from the dataset
            config: CheckConfig specifying which checks to run

        Returns:
            Dict mapping check name to pass/fail boolean.
            Empty dict if no checks are configured.
        """
        results: dict[str, bool] = {}

        if config.exact_match:
            if config.case_sensitive:
                results["exact_match"] = actual.strip() == expected.strip()
            else:
                results["exact_match"] = actual.strip().lower() == expected.strip().lower()

        for phrase in config.must_contain:
            key = f"must_contain:{phrase}"
            if config.case_sensitive:
                results[key] = phrase in actual
            else:
                results[key] = phrase.lower() in actual.lower()

        for phrase in config.must_not_contain:
            key = f"must_not_contain:{phrase}"
            if config.case_sensitive:
                results[key] = phrase not in actual
            else:
                results[key] = phrase.lower() not in actual.lower()

        if config.regex_pattern:
            try:
                results["regex_match"] = bool(re.search(config.regex_pattern, actual))
            except re.error as e:
                results["regex_match"] = False  # invalid regex = fail

        if config.custom_check:
            results["custom_check"] = HardChecker._call_custom_check(
                config.custom_check, actual, expected
            )

        return results

    @staticmethod
    def score(results: dict[str, bool]) -> float:
        """Compute aggregate hard check score.

        Returns the fraction of checks that passed.
        If no checks were configured (empty dict), returns 1.0
        (auto-pass hard checks — rely on LLM grading instead).
        """
        if not results:
            return 1.0
        return sum(1.0 for v in results.values() if v) / len(results)

    @staticmethod
    def _call_custom_check(dotted_path: str, actual: str, expected: str) -> bool:
        """Import and call a custom check function.

        The function should have the signature:
            def check(actual: str, expected: str) -> bool

        dotted_path example: "my_checks.classification.exact_category"
        """
        try:
            module_path, func_name = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return bool(func(actual, expected))
        except Exception:
            return False  # custom check error = fail
```

---

## LLM Grader Implementation

```python
import json


class LLMGrader:
    """Scores outputs using an LLM as a judge.

    Used for fuzzy/subjective tasks where deterministic checks
    are insufficient. The grading prompt is FIXED and HUMAN-DEFINED —
    the optimizer cannot modify it.

    The grading model should be DIFFERENT from the candidate model
    to prevent self-grading bias.
    """

    DEFAULT_GRADING_PROMPT = '''You are an impartial judge evaluating the quality of an AI response.

## Task Description
{task_description}

## Expected Output
{expected_output}

## Actual Output
{actual_output}

## Instructions
Rate the actual output on a scale of 0.0 to 1.0 based on how well it matches the expected output:

- 1.0: Perfect — matches the expected output in meaning and format
- 0.8-0.9: Minor differences that don't affect correctness (e.g., slight wording variation)
- 0.5-0.7: Partially correct — captures some aspects but misses important elements
- 0.2-0.4: Mostly incorrect — contains some relevant info but fundamentally wrong
- 0.0-0.1: Completely wrong or irrelevant

Respond with ONLY a JSON object in this exact format (no other text):
{{"score": <float between 0.0 and 1.0>, "explanation": "<1-2 sentence explanation>"}}'''

    def __init__(
        self,
        provider: AnthropicProvider,
        grading_model: str = "claude-sonnet-4-20250514",
        custom_grading_prompt: str | None = None,
        task_description: str = "",
    ):
        self.provider = provider
        self.grading_model = grading_model
        self.grading_prompt_template = custom_grading_prompt or self.DEFAULT_GRADING_PROMPT
        self.task_description = task_description

    async def grade(
        self,
        actual: str,
        expected: str,
    ) -> tuple[float, str]:
        """Grade an output using the LLM.

        Returns:
            (score, explanation) — score is 0.0-1.0, explanation is a short string

        If the LLM's response can't be parsed, returns (0.0, "Grading failed: ...").
        """
        prompt = self.grading_prompt_template.format(
            task_description=self.task_description,
            expected_output=expected,
            actual_output=actual,
        )

        try:
            response = await self.provider.complete(
                model=self.grading_model,
                system_prompt="You are an impartial grading judge. Respond only with valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )

            # Parse JSON from response
            result = json.loads(response.content.strip())
            score = float(result["score"])
            explanation = str(result.get("explanation", ""))

            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            return score, explanation

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return 0.0, f"Grading parse error: {str(e)}"
        except Exception as e:
            return 0.0, f"Grading failed: {str(e)}"
```

---

## Failure Clustering

```python
class FailureAnalyzer:
    """Clusters failures by pattern so the Optimizer can diagnose systematic issues.

    Uses an LLM to identify common failure patterns from a list of individual failures.
    """

    CLUSTERING_PROMPT = '''Analyze these evaluation failures and identify 2-7 common patterns.

## Failed Examples

{failures}

## Instructions
Group the failures by root cause. For each cluster:
1. Give the pattern a short, descriptive name
2. List which example IDs belong to this cluster
3. Suggest what change to the system prompt might fix this cluster

Respond with ONLY a JSON array:
[
  {{
    "pattern": "Short description of the failure pattern",
    "example_ids": ["ex_001", "ex_005"],
    "suggested_fix": "What to change in the system prompt"
  }}
]'''

    def __init__(self, provider: AnthropicProvider, model: str = "claude-sonnet-4-20250514"):
        self.provider = provider
        self.model = model

    async def cluster_failures(
        self,
        failed_scores: list[ExampleScore],
        examples: list[Example],
        example_results: list[ExampleResult],
    ) -> list[FailureCluster]:
        """Cluster failures by pattern using an LLM.

        If there are no failures or fewer than 2, returns an empty list.
        If the LLM response can't be parsed, returns a single "uncategorized" cluster.
        """
        if len(failed_scores) < 2:
            return []

        # Build failure descriptions
        failures_text = self._format_failures(failed_scores, examples, example_results)

        try:
            response = await self.provider.complete(
                model=self.model,
                system_prompt="You are a failure analysis expert. Respond only with valid JSON.",
                messages=[{"role": "user", "content": self.CLUSTERING_PROMPT.format(failures=failures_text)}],
                temperature=0.0,
                max_tokens=2000,
            )

            clusters_data = json.loads(response.content.strip())
            clusters = []
            for c in clusters_data:
                clusters.append(FailureCluster(
                    pattern=c["pattern"],
                    example_ids=c["example_ids"],
                    count=len(c["example_ids"]),
                    suggested_fix=c.get("suggested_fix", ""),
                ))
            return clusters

        except Exception:
            # Fallback: one big cluster
            return [FailureCluster(
                pattern="Uncategorized failures",
                example_ids=[s.example_id for s in failed_scores],
                count=len(failed_scores),
                suggested_fix="Review individual failures for patterns.",
            )]

    def _format_failures(
        self,
        failed_scores: list[ExampleScore],
        examples: list[Example],
        example_results: list[ExampleResult],
    ) -> str:
        """Format failures for the clustering prompt.

        Shows: example ID, input (truncated), expected, actual, failure reason.
        Limits to 30 failures to avoid context overflow.
        """
        example_map = {e.id: e for e in examples}
        result_map = {r.example_id: r for r in example_results}

        lines = []
        for score in failed_scores[:30]:  # cap at 30
            ex = example_map.get(score.example_id)
            res = result_map.get(score.example_id)
            if ex and res:
                input_preview = ex.input[:200] + ("..." if len(ex.input) > 200 else "")
                lines.append(
                    f"- ID: {score.example_id}\n"
                    f"  Input: {input_preview}\n"
                    f"  Expected: {ex.expected_output}\n"
                    f"  Actual: {res.actual_output or 'ERROR'}\n"
                    f"  Reason: {score.failure_reason}\n"
                )
        return "\n".join(lines)
```

---

## Judge (Main Class)

```python
class Judge:
    """Scores candidate outputs against expected outputs.

    Orchestrates hard checks, LLM grading, composite scoring,
    and failure clustering.
    """

    def __init__(
        self,
        provider: AnthropicProvider,
        task_config: TaskConfig,
    ):
        self.provider = provider
        self.task_config = task_config
        self.hard_checker = HardChecker()
        self.scoring = task_config.scoring

        # LLM grader — only initialized if LLM grading weight > 0
        self.llm_grader: LLMGrader | None = None
        if self.scoring.llm_grade_weight > 0:
            self.llm_grader = LLMGrader(
                provider=provider,
                grading_model=self.scoring.grading_model,
                custom_grading_prompt=self.scoring.grading_prompt,
                task_description=task_config.description,
            )

        self.failure_analyzer = FailureAnalyzer(provider)

    async def score(
        self,
        run_result: RunResult,
        examples: list[Example],
    ) -> ScoreCard:
        """Score all outputs in a RunResult.

        Process:
        1. For each example: run hard checks, optionally LLM grade, compute composite
        2. Identify failures
        3. Cluster failures by pattern
        4. Return complete ScoreCard

        Parameters:
            run_result: Raw outputs from the Runner
            examples: The original examples (needed for expected_output and check config)

        Returns:
            ScoreCard with per-example scores and failure clusters
        """
        example_map = {e.id: e for e in examples}
        example_scores: list[ExampleScore] = []

        for result in run_result.results:
            example = example_map.get(result.example_id)
            if not example:
                continue

            # Handle error cases (API errors)
            if result.error or result.actual_output is None:
                example_scores.append(ExampleScore(
                    example_id=result.example_id,
                    hard_check_score=0.0,
                    hard_check_details={},
                    llm_grade=None,
                    composite_score=0.0,
                    passed=False,
                    failure_reason=f"API error: {result.error}",
                ))
                continue

            # 1. Hard checks
            hard_results = self.hard_checker.check(
                actual=result.actual_output,
                expected=example.expected_output,
                config=example.checks,
            )
            hard_score = self.hard_checker.score(hard_results)

            # 2. LLM grading (if configured)
            llm_grade = None
            llm_explanation = ""
            if self.llm_grader and self.scoring.llm_grade_weight > 0:
                llm_grade, llm_explanation = await self.llm_grader.grade(
                    actual=result.actual_output,
                    expected=example.expected_output,
                )

            # 3. Composite score
            if llm_grade is not None:
                composite = (
                    hard_score * self.scoring.hard_check_weight
                    + llm_grade * self.scoring.llm_grade_weight
                )
            else:
                composite = hard_score  # no LLM grading = hard checks only

            # 4. Pass/fail
            passed = composite >= self.scoring.pass_threshold

            # 5. Failure reason
            failure_reason = ""
            if not passed:
                failed_checks = [k for k, v in hard_results.items() if not v]
                if failed_checks:
                    failure_reason = f"Failed hard checks: {', '.join(failed_checks)}"
                if llm_grade is not None and llm_grade < self.scoring.pass_threshold:
                    if failure_reason:
                        failure_reason += f". LLM grade: {llm_grade:.2f} — {llm_explanation}"
                    else:
                        failure_reason = f"LLM grade: {llm_grade:.2f} — {llm_explanation}"

            example_scores.append(ExampleScore(
                example_id=result.example_id,
                hard_check_score=hard_score,
                hard_check_details=hard_results,
                llm_grade=llm_grade,
                llm_grade_explanation=llm_explanation,
                composite_score=composite,
                passed=passed,
                failure_reason=failure_reason,
            ))

        # Cluster failures
        failed = [s for s in example_scores if not s.passed]
        clusters = []
        if len(failed) >= 2:
            clusters = await self.failure_analyzer.cluster_failures(
                failed_scores=failed,
                examples=examples,
                example_results=run_result.results,
            )

        return ScoreCard(
            candidate_id=run_result.candidate_id,
            run_id=run_result.run_id,
            split=run_result.split,
            example_scores=example_scores,
            failure_clusters=clusters,
        )
```

---

## Scoring Flow Summary

```
For each example in RunResult:
  │
  ├── API error? → score = 0.0, passed = False, reason = error message
  │
  ├── Run hard checks (exact_match, contains, regex, custom)
  │   └── hard_score = fraction of checks that passed (1.0 if no checks)
  │
  ├── Run LLM grader (if weight > 0)
  │   └── llm_grade = 0.0-1.0 from LLM judge
  │
  ├── composite = hard_score × hard_weight + llm_grade × soft_weight
  │
  ├── passed = composite >= pass_threshold
  │
  └── failure_reason = explanation of what went wrong

After all examples:
  │
  ├── Cluster failures by pattern (LLM-assisted)
  │
  └── Return ScoreCard with all scores + clusters
```

---

## Storage Format

ScoreCard saved as JSON:
```json
{
  "candidate_id": "candidate_003",
  "run_id": "run_20260322_143000",
  "split": "train",
  "scored_at": "2026-03-22T14:35:00Z",
  "example_scores": [
    {
      "example_id": "ex_001",
      "hard_check_score": 1.0,
      "hard_check_details": {"exact_match": true},
      "llm_grade": null,
      "composite_score": 1.0,
      "passed": true,
      "failure_reason": ""
    },
    {
      "example_id": "ex_002",
      "hard_check_score": 0.0,
      "hard_check_details": {"exact_match": false},
      "llm_grade": 0.3,
      "composite_score": 0.09,
      "passed": false,
      "failure_reason": "Failed hard checks: exact_match. LLM grade: 0.30 — Output included extra explanation."
    }
  ],
  "failure_clusters": [
    {
      "pattern": "Model adds explanation text after the category name",
      "example_ids": ["ex_002", "ex_007", "ex_015"],
      "count": 3,
      "representative_example": "",
      "suggested_fix": "Add 'Respond with ONLY the category name, no explanation' to the system prompt"
    }
  ]
}
```

---

## Testing Requirements

```
# Hard checks
test_hard_check_exact_match_pass
test_hard_check_exact_match_fail
test_hard_check_case_insensitive
test_hard_check_must_contain_pass
test_hard_check_must_contain_fail
test_hard_check_must_not_contain_pass
test_hard_check_must_not_contain_fail
test_hard_check_regex_match_pass
test_hard_check_regex_match_fail
test_hard_check_regex_invalid_pattern
test_hard_check_custom_check_pass
test_hard_check_custom_check_fail
test_hard_check_no_checks_returns_empty_dict
test_hard_check_score_all_pass
test_hard_check_score_all_fail
test_hard_check_score_mixed
test_hard_check_score_empty_dict_returns_one

# LLM grader
test_llm_grader_high_score
test_llm_grader_low_score
test_llm_grader_parse_error_returns_zero
test_llm_grader_api_error_returns_zero
test_llm_grader_clamps_score_to_range
test_llm_grader_uses_correct_model  # must differ from candidate

# Composite scoring
test_composite_hard_only_when_no_llm_weight
test_composite_weighted_combination
test_composite_pass_at_threshold
test_composite_fail_below_threshold

# Failure clustering
test_cluster_failures_groups_similar
test_cluster_fewer_than_2_returns_empty
test_cluster_parse_error_returns_fallback

# Judge (integration)
test_judge_score_all_pass
test_judge_score_all_fail
test_judge_score_mixed
test_judge_handles_api_errors_in_results
test_judge_skips_llm_grading_when_weight_zero
test_judge_produces_failure_clusters

# ScoreCard properties
test_scorecard_accuracy
test_scorecard_pass_rate
test_scorecard_total_passed_failed
test_scorecard_empty

# Serialization
test_scorecard_json_roundtrip
test_example_score_json_roundtrip
```

Mock ALL LLM API calls. Use `pytest.mark.asyncio` for async tests.

**Minimum: 30 test cases.**

---

## Implementation Notes

- The Judge is async because LLM grading makes API calls
- Hard checks run synchronously (they're fast, no I/O)
- LLM grading runs sequentially per example (to avoid rate limit issues on the grading model)
- The grading model should ideally be different from the candidate model. If they're the same, log a warning.
- The failure clustering prompt limits input to 30 failures to avoid context overflow
- Custom check functions are imported dynamically — they must be importable from the Python path
- The Judge does NOT know about the Optimizer. It just produces ScoreCards. The Controller passes them to the Optimizer.

---

## Dependencies

- **Depends on:** Ticket 1 (config), Ticket 2 (Example, CheckConfig, TaskConfig, ScoringConfig), Ticket 3 (Candidate — for candidate_id), Ticket 4 (RunResult, ExampleResult, AnthropicProvider)
- **Depended on by:** Ticket 6 (Optimizer reads ScoreCards and FailureClusters), Ticket 7 (Controller uses Judge to score each round)
