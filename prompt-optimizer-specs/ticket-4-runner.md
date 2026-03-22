# Ticket 4: Runner (LLM Execution Engine)

## Overview

The Runner takes a candidate configuration and a list of examples, calls the Anthropic API for each input, and collects raw outputs. This is the component that actually spends money on API calls, so it needs proper rate limiting, retries, cost tracking, and parallel execution.

The only provider for v1 is **Anthropic (Claude)**. The Runner uses async execution (asyncio) with configurable concurrency.

---

## Acceptance Criteria

- [ ] `Runner.run(candidate, examples, split_name) -> RunResult` executes all examples
- [ ] Async execution using `asyncio` for parallel API calls
- [ ] Configurable concurrency limit via `asyncio.Semaphore` (default: 10 concurrent requests)
- [ ] Exponential backoff retry on rate limits (429) and server errors (500, 502, 503, 529)
- [ ] Max 3 retries per request, backoff: 2s, 4s, 8s
- [ ] Do NOT retry on client errors: 400 (bad request), 401 (auth), 404 (not found)
- [ ] Per-example tracking: `input_tokens`, `output_tokens`, `latency_ms`, `actual_output`, `error`
- [ ] Aggregate tracking: `total_tokens`, `estimated_cost_usd`, `avg_latency_ms`, `success_rate`
- [ ] Anthropic provider using the official `anthropic` Python SDK (async client)
- [ ] System prompt sent via the `system` parameter (not as a user message)
- [ ] Few-shot examples injected as alternating user/assistant message pairs before the actual input
- [ ] Progress bar during execution using `rich.progress`
- [ ] Graceful handling of API errors (timeout, auth failure, invalid request)
- [ ] Failed examples recorded with error message (not silently dropped)
- [ ] Results saved to disk as JSONL

---

## Data Models

```python
from pydantic import BaseModel, Field
from datetime import datetime


class ExampleResult(BaseModel):
    """Result of running a single example through the candidate.

    Captures the model's raw output, token usage, latency, and any errors.
    """
    example_id: str
    input: str
    expected_output: str
    actual_output: str | None = None   # None if the call errored
    error: str | None = None           # Error message if the call failed
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class RunResult(BaseModel):
    """Aggregate result of running a candidate against a set of examples.

    Contains per-example results and aggregate metrics.
    """
    candidate_id: str
    run_id: str                        # Auto-generated: "run_YYYYMMDD_HHMMSS"
    split: str                         # "train", "validation", or "test"
    started_at: datetime
    completed_at: datetime
    results: list[ExampleResult]

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all examples."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on model pricing.

        Uses the pricing table from AnthropicProvider.
        Note: This is an estimate. Actual billing may differ.
        """
        # Implementation delegates to the provider's pricing table
        ...

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.actual_output is not None)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.success_count / len(self.results)

    @property
    def avg_latency_ms(self) -> float:
        successful = [r.latency_ms for r in self.results if r.actual_output is not None]
        if not successful:
            return 0.0
        return sum(successful) / len(successful)
```

---

## Anthropic Provider

```python
import anthropic
import time
from typing import Any


# Pricing per 1M tokens (input, output) in USD
ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-opus-4-20250514": (15.0, 75.0),
}


class ProviderResponse(BaseModel):
    """Structured response from an LLM API call."""
    content: str                   # The model's text response
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: int


class AnthropicProvider:
    """Wrapper around the Anthropic async API client.

    Handles API calls, token tracking, and cost estimation.
    """

    def __init__(self, api_key: str, default_model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.default_model = default_model

    async def complete(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        """Call the Anthropic Messages API and return a structured response.

        Parameters:
            model: Model ID (e.g., "claude-sonnet-4-20250514")
            system_prompt: System prompt (sent via the `system` parameter)
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens

        Returns:
            ProviderResponse with content, token counts, and latency

        Raises:
            anthropic.APIError: On API errors (after retries are exhausted)
        """
        start = time.monotonic()

        response = await self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = int((time.monotonic() - start) * 1000)

        return ProviderResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            latency_ms=latency_ms,
        )

    @staticmethod
    def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a given model and token count.

        Returns 0.0 if the model is not in the pricing table.
        """
        if model not in ANTHROPIC_PRICING:
            return 0.0
        input_price, output_price = ANTHROPIC_PRICING[model]
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.close()
```

---

## Runner Implementation

```python
import asyncio
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn


class Runner:
    """Executes a candidate configuration against a set of eval examples.

    Manages concurrency, retries, progress display, and result collection.
    """

    # Retry configuration
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
    MAX_RETRIES = 3
    RETRY_BACKOFF = [2, 4, 8]  # seconds

    def __init__(self, provider: AnthropicProvider, max_concurrent: int = 10):
        self.provider = provider
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run(
        self,
        candidate: Candidate,
        examples: list[Example],
        split_name: str = "train",
        dataset: Dataset | None = None,
    ) -> RunResult:
        """Execute the candidate against all examples with concurrency control.

        Parameters:
            candidate: The candidate config to test
            examples: List of examples to run
            split_name: Which split these examples come from ("train", "validation", "test")
            dataset: Optional dataset reference for resolving few-shot example IDs

        Returns:
            RunResult with per-example results and aggregate metrics
        """
        started_at = datetime.utcnow()
        run_id = f"run_{started_at.strftime('%Y%m%d_%H%M%S')}"

        # Resolve few-shot examples
        few_shot_messages = self._build_few_shot_messages(candidate, dataset)

        # Run all examples with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("errors: {task.fields[errors]}"),
        ) as progress:
            task = progress.add_task(
                f"Running {candidate.id} on {split_name}",
                total=len(examples),
                errors=0,
            )
            error_count = 0

            async def run_with_progress(example: Example) -> ExampleResult:
                nonlocal error_count
                result = await self._run_single(candidate, example, few_shot_messages)
                if result.error:
                    error_count += 1
                progress.update(task, advance=1, errors=error_count)
                return result

            tasks = [run_with_progress(ex) for ex in examples]
            results = await asyncio.gather(*tasks)

        completed_at = datetime.utcnow()

        return RunResult(
            candidate_id=candidate.id,
            run_id=run_id,
            split=split_name,
            started_at=started_at,
            completed_at=completed_at,
            results=list(results),
        )

    async def _run_single(
        self,
        candidate: Candidate,
        example: Example,
        few_shot_messages: list[dict[str, str]],
    ) -> ExampleResult:
        """Execute a single example with concurrency control and retry logic.

        Retries on transient errors (429, 5xx) with exponential backoff.
        Does NOT retry on client errors (400, 401, 404).
        """
        async with self.semaphore:
            messages = self._build_messages(example, few_shot_messages)

            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    response = await self.provider.complete(
                        model=candidate.model,
                        system_prompt=candidate.system_prompt,
                        messages=messages,
                        temperature=candidate.temperature,
                        max_tokens=candidate.max_tokens,
                    )
                    return ExampleResult(
                        example_id=example.id,
                        input=example.input,
                        expected_output=example.expected_output,
                        actual_output=response.content,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        latency_ms=response.latency_ms,
                    )
                except anthropic.APIStatusError as e:
                    if e.status_code in self.RETRYABLE_STATUS_CODES and attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_BACKOFF[attempt])
                        continue
                    return ExampleResult(
                        example_id=example.id,
                        input=example.input,
                        expected_output=example.expected_output,
                        error=f"API error {e.status_code}: {e.message}",
                    )
                except anthropic.APIConnectionError as e:
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_BACKOFF[attempt])
                        continue
                    return ExampleResult(
                        example_id=example.id,
                        input=example.input,
                        expected_output=example.expected_output,
                        error=f"Connection error: {str(e)}",
                    )
                except Exception as e:
                    return ExampleResult(
                        example_id=example.id,
                        input=example.input,
                        expected_output=example.expected_output,
                        error=f"Unexpected error: {str(e)}",
                    )

    def _build_messages(
        self,
        example: Example,
        few_shot_messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Build the messages array for the API call.

        Structure:
        1. Few-shot examples (if any) as alternating user/assistant pairs
        2. The actual input as the final user message

        The system prompt is NOT included here — it goes in the `system` parameter.
        """
        messages = list(few_shot_messages)  # copy
        messages.append({"role": "user", "content": example.input})
        return messages

    def _build_few_shot_messages(
        self,
        candidate: Candidate,
        dataset: Dataset | None,
    ) -> list[dict[str, str]]:
        """Build few-shot example messages from candidate's few_shot_example_ids.

        Each few-shot example becomes two messages:
        - {"role": "user", "content": example.input}
        - {"role": "assistant", "content": example.expected_output}

        Returns empty list if no few-shot examples configured.
        """
        if not candidate.few_shot_example_ids or dataset is None:
            return []

        messages = []
        for example_id in candidate.few_shot_example_ids:
            example = dataset.get_example_by_id(example_id)
            if example:
                messages.append({"role": "user", "content": example.input})
                messages.append({"role": "assistant", "content": example.expected_output})
        return messages
```

---

## Message Construction Detail

The Anthropic Messages API takes:
- `system` parameter: the system prompt (string)
- `messages` parameter: list of user/assistant turns

For a candidate with 2 few-shot examples and one actual input:

```python
# API call structure:
await client.messages.create(
    model="claude-sonnet-4-20250514",
    system="You are a ticket classifier...",     # ← system_prompt goes here
    messages=[
        # Few-shot example 1
        {"role": "user", "content": "My order is late"},
        {"role": "assistant", "content": "shipping"},
        # Few-shot example 2
        {"role": "user", "content": "Charged twice"},
        {"role": "assistant", "content": "billing"},
        # Actual input (always last)
        {"role": "user", "content": "Package arrived broken"},
    ],
    temperature=0.0,
    max_tokens=50,
)
```

---

## Retry Logic

| Status Code | Meaning | Retry? | Notes |
|------------|---------|--------|-------|
| 429 | Rate limited | YES | Backoff: 2s, 4s, 8s |
| 500 | Server error | YES | Backoff: 2s, 4s, 8s |
| 502 | Bad gateway | YES | Backoff: 2s, 4s, 8s |
| 503 | Service unavailable | YES | Backoff: 2s, 4s, 8s |
| 529 | Overloaded | YES | Backoff: 2s, 4s, 8s |
| 400 | Bad request | NO | Log error, return immediately |
| 401 | Auth failed | NO | Log error, return immediately |
| 404 | Not found | NO | Log error, return immediately |

Max retries: 3 (so up to 4 total attempts).
After all retries exhausted: return `ExampleResult` with `error` set and `actual_output=None`.

---

## Cost Tracking

### Pricing Table (as of 2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| claude-sonnet-4-20250514 | $3.00 | $15.00 |
| claude-haiku-4-5-20251001 | $0.80 | $4.00 |
| claude-opus-4-20250514 | $15.00 | $75.00 |

### Calculation

```python
cost_usd = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
```

The `RunResult.estimated_cost_usd` property should use the candidate's model to look up pricing and sum across all example results.

---

## Progress Display

During execution, show:
```
Running candidate_003 on train...
[████████████████░░░░░░░░░░░░░░] 120/200  errors: 2
```

Use `rich.progress.Progress` with:
- `SpinnerColumn()` — spinning indicator
- `TextColumn` — description
- `BarColumn()` — progress bar
- `MofNCompleteColumn()` — "120/200"
- Custom field for error count

---

## Output Storage

Results are saved as JSONL to: `runs/<run_id>/rounds/round_NNN/<candidate_id>_<split>_results.jsonl`

Each line is one `ExampleResult` serialized as JSON:
```jsonl
{"example_id": "ex_001", "input": "My order is late", "expected_output": "shipping", "actual_output": "shipping", "error": null, "input_tokens": 85, "output_tokens": 3, "latency_ms": 340}
{"example_id": "ex_002", "input": "Charged twice", "expected_output": "billing", "actual_output": "billing_issue", "error": null, "input_tokens": 78, "output_tokens": 4, "latency_ms": 290}
```

Provide save/load utilities:
```python
def save_run_result(result: RunResult, path: Path) -> None:
    """Save RunResult to JSONL file."""
    with open(path, "w") as f:
        for example_result in result.results:
            f.write(example_result.model_dump_json() + "\n")

def load_run_result(path: Path, metadata: dict) -> RunResult:
    """Load RunResult from JSONL file + metadata."""
    ...
```

---

## Testing Requirements

All tests must mock the Anthropic API — no real API calls in tests.

```
# Message construction
test_build_messages_no_few_shot
test_build_messages_with_few_shot
test_build_messages_few_shot_order_preserved
test_build_messages_actual_input_is_last

# Retry logic
test_retry_on_429_then_success
test_retry_on_500_then_success
test_retry_exhausted_returns_error
test_no_retry_on_400
test_no_retry_on_401
test_retry_backoff_timing

# Concurrency
test_concurrent_execution_respects_semaphore
test_concurrent_execution_all_complete

# Provider
test_provider_complete_success
test_provider_complete_extracts_content
test_provider_complete_tracks_tokens
test_provider_complete_tracks_latency
test_provider_connection_error

# Cost estimation
test_estimate_cost_sonnet
test_estimate_cost_haiku
test_estimate_cost_opus
test_estimate_cost_unknown_model

# RunResult properties
test_run_result_total_tokens
test_run_result_success_count
test_run_result_error_count
test_run_result_success_rate
test_run_result_avg_latency
test_run_result_estimated_cost

# Serialization
test_save_run_result_jsonl
test_load_run_result_jsonl

# Error handling
test_run_single_timeout
test_run_single_unexpected_error
test_run_result_with_mixed_success_and_errors
```

Use `unittest.mock.AsyncMock` to mock async API calls.
Use `pytest.mark.asyncio` for async tests.

**Minimum: 25 test cases.**

---

## Implementation Notes

- Use `anthropic.AsyncAnthropic` (not the sync client) — the entire Runner is async
- The `semaphore` limits concurrency to prevent overwhelming the API
- `asyncio.gather` runs all examples concurrently (within semaphore limits)
- Time tracking uses `time.monotonic()` for accurate latency measurement
- The progress bar updates in real-time as examples complete
- If ALL examples fail (e.g., auth error), the RunResult still contains all results with error messages
- The Runner has no opinion on scoring — it just collects raw outputs. Scoring is the Judge's job.

---

## Dependencies

- **Depends on:** Ticket 1 (config, API key from Settings), Ticket 2 (Dataset, Example model), Ticket 3 (Candidate model)
- **Depended on by:** Ticket 5 (Judge scores RunResults), Ticket 7 (Controller calls Runner each round)
