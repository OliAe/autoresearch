# Epic: Eval-Driven LLM Program Optimizer (v1)

## Vision

A system where you provide input/output examples, and an autonomous loop
iterates on the system prompt and model choice until accuracy converges —
no manual prompt engineering required.

You define **what good looks like**. The system figures out **how to get there**.

---

## Core Loop

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│   │  Runner   │───▶│  Judge   │───▶│    Optimizer     │  │
│   │          │    │          │    │                  │  │
│   │ Executes │    │ Scores   │    │ Reviews failures │  │
│   │ candidate│    │ each     │    │ Proposes next    │  │
│   │ against  │    │ output   │    │ candidate        │  │
│   │ eval set │    │ vs ideal │    │ (prompt, model)  │  │
│   └──────────┘    └──────────┘    └────────┬─────────┘  │
│        ▲                                   │            │
│        │           ┌──────────┐            │            │
│        └───────────│Controller│◀───────────┘            │
│                    │          │                          │
│                    │ Tracks   │                          │
│                    │ champion │                          │
│                    │ history  │                          │
│                    │ stopping │                          │
│                    └──────────┘                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### One iteration, spelled out:

1. **Controller** picks the current candidate config (system prompt + model + temperature)
2. **Runner** sends every input from the eval set through the candidate config, collects raw outputs
3. **Judge** scores each output against the expected output using hard checks first, then an LLM grader for fuzzy cases
4. **Optimizer** receives the full scorecard (which inputs passed, which failed, why), analyzes failure patterns, and proposes a new candidate
5. **Controller** compares the new candidate's score to the current champion — keeps the better one, logs the result, checks stopping criteria

---

## Architecture

### 1. Dataset Store

The foundation. You fill this with your examples.

```
datasets/
  my_task/
    config.yaml          # task metadata, scoring weights, model allowlist
    examples.jsonl       # all input/output pairs
    splits/
      train.jsonl        # optimizer sees these (70%)
      validation.jsonl   # used to pick champion (15%)
      test.jsonl         # frozen, never touched during optimization (15%)
```

Each example:
```json
{
  "id": "ex_001",
  "input": "Classify this support ticket: 'My order arrived damaged'",
  "expected_output": "product_damage",
  "metadata": {
    "difficulty": "easy",
    "category": "classification"
  },
  "checks": {
    "exact_match": true,
    "case_sensitive": false,
    "must_contain": [],
    "must_not_contain": [],
    "json_schema": null,
    "custom_check": null
  }
}
```

**CLI to populate:**
```bash
# Add examples one at a time
prompt-opt add --task my_task --input "..." --output "..."

# Bulk import from CSV/JSON
prompt-opt import --task my_task --file examples.csv

# Auto-split into train/val/test
prompt-opt split --task my_task --train 0.7 --val 0.15 --test 0.15
```

### 2. Candidate Representation

A candidate is everything the optimizer can change, frozen into a config:

```yaml
# candidates/candidate_042.yaml
model: "claude-sonnet-4-20250514"
system_prompt: |
  You are a customer support ticket classifier.
  Classify each ticket into exactly one category.
  Categories: billing, product_damage, shipping, account, other.
  Respond with only the category name, nothing else.
temperature: 0.0
max_tokens: 50
few_shot_examples: []        # v1: empty or list of example IDs
```

**v1 search space (what the optimizer can change):**

| Knob | Type | Range |
|------|------|-------|
| `system_prompt` | string | free text |
| `model` | enum | from allowlist in task config |
| `temperature` | float | 0.0 – 1.0 |
| `max_tokens` | int | 1 – 4096 |
| `few_shot_examples` | list[id] | subset of train examples |

### 3. Runner

Executes a candidate config against a set of examples.

**Responsibilities:**
- Call the LLM API with the candidate's model + system prompt + each input
- Collect raw outputs
- Handle retries, rate limits, timeouts
- Log token usage and latency per call
- Support parallel execution (batch inputs for speed)

**Output:**
```json
{
  "candidate_id": "candidate_042",
  "run_id": "run_20260322_143000",
  "results": [
    {
      "example_id": "ex_001",
      "input": "...",
      "expected_output": "product_damage",
      "actual_output": "product_damage",
      "latency_ms": 340,
      "tokens_in": 85,
      "tokens_out": 3
    }
  ]
}
```

### 4. Judge

Scores each output. Two layers:

**Layer 1 — Hard checks (deterministic, fast, trustworthy):**
- Exact match
- Contains / not-contains
- Regex match
- JSON schema validation
- Field-level extraction checks
- Custom Python functions

**Layer 2 — LLM grader (for fuzzy/subjective tasks):**
- Only runs when hard checks are insufficient (e.g., open-ended generation)
- Uses a *different model* than the candidate being evaluated (prevent self-grading)
- Grading prompt is fixed and human-defined — the optimizer cannot change it
- Returns a score (0-1) and a short explanation

**Composite score per example:**
```
score = hard_check_score × hard_weight + llm_grade × soft_weight
```

**Aggregate score per candidate:**
```
accuracy    = mean(score across all examples)
pass_rate   = count(score >= threshold) / total
cost        = total tokens × price_per_token
avg_latency = mean(latency_ms)

final_score = accuracy - cost_penalty - latency_penalty
```

**Important rule:** The judge's grading prompt and scoring weights are human-defined and locked. The optimizer cannot modify them. This prevents reward hacking.

### 5. Optimizer

The "brain" — an LLM that analyzes results and proposes improvements.

**Input it receives each round:**
- Current candidate config
- Per-example scores and explanations
- Failure clusters (grouped by error type)
- History of previous candidates and their scores
- The task config (what models are allowed, what the task is)

**What it does:**
1. Clusters failures: "12 examples failed because the model returned explanations instead of just the category name"
2. Diagnoses root cause: "The system prompt says 'classify' but doesn't say 'respond with only the category name'"
3. Proposes a fix: new system prompt, or model switch, or temperature change
4. Outputs a new candidate config

**What it cannot do:**
- Change the scoring function
- Change the expected outputs
- Change the judge's grading prompt
- Deploy anything to production
- Access the test set

**Optimizer prompt structure:**
```
You are an LLM program optimizer.

Your job: analyze evaluation results and propose a better candidate config
(system prompt, model, temperature, few-shot examples) that will score
higher on the next round.

## Current candidate
{candidate_config}

## Results summary
Overall accuracy: 78%
Pass rate: 72%
Total examples: 200

## Failure analysis
Cluster 1 (15 failures): Model adds explanation text after the category
Cluster 2 (8 failures): Model outputs "damaged_product" instead of "product_damage"
Cluster 3 (5 failures): Model says "I'm not sure" instead of picking a category

## Previous attempts
Round 1: accuracy 65% (baseline prompt)
Round 2: accuracy 72% (added "respond with only the category")
Round 3: accuracy 78% (added explicit category list)  ← current champion

## Constraints
- Allowed models: [claude-sonnet-4-20250514, claude-haiku-4-5-20251001, gpt-4o, gpt-4o-mini]
- Temperature range: 0.0 - 1.0
- Max system prompt length: 2000 tokens

Propose a new candidate config. Explain your reasoning, then output the
config in the specified YAML format.
```

### 6. Controller

The outer loop that ties everything together.

**Responsibilities:**
- Initialize with baseline candidate (minimal prompt, default model)
- Run the loop: runner → judge → optimizer → repeat
- Track champion (best candidate on validation set)
- Track full history (every candidate, every score)
- Implement stopping criteria:
  - Accuracy plateau (< epsilon improvement over N rounds)
  - Budget exhausted (max rounds or max API spend)
  - Perfect score reached
  - Overfitting detected (train score rising, validation score flat/falling)
- Final evaluation on frozen test set (once, at the end)
- Generate report

**Loop pseudocode:**
```python
champion = baseline_candidate()
history = []

for round in range(max_rounds):
    # 1. Run candidate on train split
    outputs = runner.run(champion, dataset.train)

    # 2. Score outputs
    scores = judge.score(outputs)

    # 3. Log
    history.append({"round": round, "candidate": champion, "scores": scores})

    # 4. Check stopping criteria
    if scores.accuracy >= target_accuracy:
        break
    if plateaued(history, patience=5):
        break

    # 5. Optimizer proposes next candidate
    challenger = optimizer.propose(
        current=champion,
        scores=scores,
        history=history,
        task_config=task_config,
    )

    # 6. Run challenger on train split
    challenger_outputs = runner.run(challenger, dataset.train)
    challenger_scores = judge.score(challenger_outputs)

    # 7. Validate on validation split
    val_scores_champion = judge.score(runner.run(champion, dataset.validation))
    val_scores_challenger = judge.score(runner.run(challenger, dataset.validation))

    # 8. Promote if better
    if val_scores_challenger.final > val_scores_champion.final:
        champion = challenger
        print(f"Round {round}: New champion! {val_scores_challenger.final:.3f}")
    else:
        print(f"Round {round}: Champion holds. {val_scores_champion.final:.3f}")

# Final test
test_scores = judge.score(runner.run(champion, dataset.test))
report(champion, test_scores, history)
```

---

## Project Structure

```
prompt-optimizer/
├── pyproject.toml
├── README.md
├── src/
│   └── prompt_optimizer/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point
│       ├── dataset.py          # Dataset loading, splitting, validation
│       ├── candidate.py        # Candidate config representation
│       ├── runner.py           # LLM API execution engine
│       ├── judge.py            # Hard checks + LLM grader
│       ├── optimizer.py        # Failure analysis + candidate proposal
│       ├── controller.py       # Main loop, champion tracking, stopping
│       ├── history.py          # Run history, logging, reporting
│       └── providers/
│           ├── __init__.py
│           ├── anthropic.py    # Claude API wrapper
│           └── openai.py       # OpenAI API wrapper
├── datasets/                   # User's eval datasets live here
│   └── example_task/
│       ├── config.yaml
│       └── examples.jsonl
├── runs/                       # Output from optimization runs
└── tests/
    ├── test_dataset.py
    ├── test_judge.py
    ├── test_runner.py
    └── test_optimizer.py
```

---

## CLI Interface

```bash
# Setup a new task
prompt-opt init my_task --description "Classify support tickets"

# Add examples
prompt-opt add my_task --input "My order is late" --output "shipping"
prompt-opt import my_task --file examples.csv

# Split dataset
prompt-opt split my_task --train 0.7 --val 0.15 --test 0.15

# Configure allowed models
prompt-opt config my_task --models "claude-sonnet-4-20250514,gpt-4o-mini"

# Run optimization
prompt-opt run my_task --max-rounds 50 --target-accuracy 0.95

# Check status of a running optimization
prompt-opt status my_task

# View results
prompt-opt results my_task
prompt-opt report my_task --format markdown

# Export the winning config
prompt-opt export my_task --output best_config.yaml
```

---

## What Each Subtask Ticket Should Cover

### Ticket 1: Dataset store and CLI
- Example schema (JSONL)
- Task config schema (YAML)
- `init`, `add`, `import`, `split` commands
- Validation (no duplicate IDs, required fields, split integrity)
- Test set isolation (warn if accessed outside final eval)

### Ticket 2: Candidate representation
- YAML config schema
- Validation (model must be in allowlist, temperature in range, etc.)
- Baseline candidate generator (minimal prompt from task description)
- Candidate diff/comparison utilities

### Ticket 3: Runner (LLM execution engine)
- Multi-provider support (Anthropic, OpenAI)
- Parallel execution with rate limiting
- Retry logic with exponential backoff
- Token usage and latency tracking per call
- Structured output collection

### Ticket 4: Judge (scoring engine)
- Hard check framework (exact match, contains, regex, JSON schema, custom)
- LLM grader with fixed human-defined grading prompt
- Composite scoring (hard × weight + soft × weight)
- Aggregate metrics (accuracy, pass rate, cost, latency)
- Per-example explanations for failures
- Failure clustering (group similar errors)

### Ticket 5: Optimizer (proposal engine)
- Failure analysis prompt construction
- History summarization (keep context manageable)
- Candidate proposal with reasoning
- Constraint enforcement (allowed models, prompt length limits)
- Staged search: prompt changes first, model changes only after plateau

### Ticket 6: Controller (main loop)
- Champion/challenger evaluation
- Validation-based promotion
- Stopping criteria (plateau, budget, target, overfitting)
- Run logging and history persistence
- Final test-set evaluation
- Report generation

### Ticket 7: CLI and configuration
- Click/Typer CLI wiring
- Global config (API keys, default providers, cost budgets)
- `run`, `status`, `results`, `report`, `export` commands
- Progress display during optimization

---

## Non-Goals for v1

- Ensembles / voting — deferred to v2
- Routing between models per-input — deferred to v2
- Fine-tuning integration — deferred to v3
- Production deployment pipeline — deferred to v3
- Web UI — deferred, CLI only for now
- Retrieval/RAG as a knob — deferred to v2

---

## Key Design Rules

1. **The optimizer never modifies the judge.** Scoring criteria are human-owned.
2. **The test set is frozen.** Only touched once, at the very end.
3. **Validation decides promotion.** Train set is for the optimizer to learn from; validation set picks the champion.
4. **Hard checks before soft checks.** Deterministic scoring wherever possible. LLM grading only for genuinely subjective tasks.
5. **History is immutable.** Every candidate, every score, every round is logged. No silent overwrites.
6. **Cost is part of the score.** A candidate that uses an expensive model gets penalized unless it's meaningfully better.
7. **Staged search.** Prompt and few-shot changes first. Model switching only after prompt search plateaus.

---

## Success Criteria

- Given 50+ input/output examples for a well-defined task, the system should find a prompt+model config that scores within 5% of a human-crafted expert prompt within 20 optimization rounds
- The system should plateau gracefully, not oscillate
- Cost of the optimization run should be trackable and bounded
- The winning config should be exportable as a standalone YAML that any developer can use directly
