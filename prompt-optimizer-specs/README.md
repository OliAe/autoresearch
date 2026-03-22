# Prompt Optimizer — Eval-Driven LLM Program Optimization

## What This Is

A system where you provide input/output example pairs, and an automated loop iterates on system prompts and model choices until accuracy converges. No manual prompt engineering required.

**You define what good looks like. The system figures out how to get there.**

Instead of manually tweaking prompts, running them in your head, and tweaking again, this system:
1. Takes your input/output examples (your eval dataset)
2. Runs candidates (system prompt + model + params) against them
3. Scores every output automatically
4. Uses an LLM to analyze failures and propose improvements
5. Repeats until accuracy converges or budget is exhausted

The default and only provider for v1 is **Anthropic (Claude)**.

---

## The Core Problem

Prompt engineering today is manual trial-and-error:
- Write a prompt → test it on a few examples mentally → notice a failure → tweak the prompt → repeat
- No structured scoring — you're eyeballing results
- No history — you forget what you already tried
- No separation of training vs. validation — you overfit to the examples you're looking at
- No budget tracking — you have no idea how much you've spent

This system automates the entire loop with structured evaluation, proper train/val/test splits, and a budget-aware optimization loop.

---

## How It Works — The Core Loop

```
┌──────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LOOP                         │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────────┐   │
│   │  RUNNER   │───▶│  JUDGE   │───▶│     OPTIMIZER        │   │
│   │          │    │          │    │                      │   │
│   │ Executes │    │ Scores   │    │ Analyzes failures    │   │
│   │ candidate│    │ each     │    │ Proposes next        │   │
│   │ against  │    │ output   │    │ candidate config     │   │
│   │ eval set │    │ vs ideal │    │ (prompt/model/temp)  │   │
│   └──────────┘    └──────────┘    └────────┬─────────────┘   │
│        ▲                                   │                 │
│        │           ┌──────────┐            │                 │
│        └───────────│CONTROLLER│◀───────────┘                 │
│                    │          │                               │
│                    │ Tracks   │                               │
│                    │ champion │                               │
│                    │ history  │                               │
│                    │ stopping │                               │
│                    └──────────┘                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### One iteration, spelled out:

1. **Controller** picks the current candidate config (system prompt + model + temperature)
2. **Runner** sends every input from the eval set through the candidate config, collects raw outputs via the Anthropic API
3. **Judge** scores each output against the expected output — deterministic hard checks first (exact match, regex, contains), then an LLM grader for fuzzy/subjective tasks
4. **Optimizer** receives the full scorecard (which inputs passed, which failed, failure patterns clustered by type), analyzes root causes, and proposes a new candidate
5. **Controller** runs the challenger on the validation set, compares to the current champion — keeps the better one, logs the result, checks stopping criteria

### The Three Model Roles

| Role | What It Does | Can It Be Changed By The Optimizer? |
|------|-------------|-------------------------------------|
| **Candidate** (Runner) | The model being tested/optimized | YES — this is the target |
| **Grader** (Judge) | Scores fuzzy outputs against expected | NO — human-defined, locked |
| **Brain** (Optimizer) | Analyzes failures, proposes improvements | NO — fixed role |

The **critical rule**: the Optimizer can change the candidate's prompt and model, but **never** the Judge's scoring criteria. This prevents reward hacking.

---

## Key Design Principles

1. **Optimizer never modifies the Judge** — Scoring criteria are human-owned and locked
2. **Test set is frozen** — Only touched once, at the very end, for final evaluation
3. **Validation decides promotion** — Train set is for learning, validation set picks the champion
4. **Hard checks before soft checks** — Deterministic scoring wherever possible; LLM grading only for genuinely subjective tasks
5. **History is immutable** — Every candidate, every score, every round is logged; no overwrites
6. **Cost is part of the score** — A candidate using an expensive model gets penalized unless it's meaningfully better
7. **Staged search** — Prompt and few-shot changes first; model switching only after prompt search plateaus

---

## Project Structure

```
prompt-optimizer/
├── pyproject.toml                  # Package config, dependencies, CLI entry point
├── README.md                       # This file
├── .env.example                    # Template for API keys
├── src/
│   └── prompt_optimizer/
│       ├── __init__.py             # Package version, top-level exports
│       ├── cli.py                  # Click CLI — all user-facing commands
│       ├── config.py               # Global settings (Pydantic BaseSettings)
│       ├── dataset.py              # Dataset loading, splitting, validation
│       ├── candidate.py            # Candidate config representation & factory
│       ├── runner.py               # LLM API execution engine
│       ├── judge.py                # Hard checks + LLM grader + failure clustering
│       ├── optimizer.py            # Failure analysis + candidate proposal
│       ├── controller.py           # Main loop, champion tracking, stopping
│       ├── history.py              # Run history, logging, persistence, reporting
│       └── providers/
│           ├── __init__.py
│           └── anthropic.py        # Anthropic Claude API wrapper
├── datasets/                       # User's eval datasets (created by CLI)
│   └── example_task/
│       ├── config.yaml             # Task metadata, allowed models, scoring config
│       ├── examples.jsonl          # All input/output pairs
│       └── splits/
│           ├── train.jsonl         # 70% — optimizer learns from these
│           ├── validation.jsonl    # 15% — decides champion promotion
│           └── test.jsonl          # 15% — frozen, final eval only
├── runs/                           # Output from optimization runs
│   └── run_20260322_143000/
│       ├── config.yaml             # Snapshot of run config
│       ├── rounds/                 # Per-round data
│       │   ├── round_001/
│       │   │   ├── candidate.yaml
│       │   │   ├── train_results.jsonl
│       │   │   ├── train_scores.json
│       │   │   ├── validation_results.jsonl
│       │   │   ├── validation_scores.json
│       │   │   └── proposal.json
│       │   └── round_002/
│       │       └── ...
│       ├── champion.yaml           # Final champion config
│       ├── report.md               # Human-readable report
│       └── summary.json            # Machine-readable summary
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Shared fixtures
    ├── test_dataset.py
    ├── test_candidate.py
    ├── test_runner.py
    ├── test_judge.py
    ├── test_optimizer.py
    ├── test_controller.py
    └── test_cli.py
```

---

## User Journey

### 1. Install
```bash
git clone <repo>
cd prompt-optimizer
uv sync
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Create a Task
```bash
prompt-opt init my_classifier --description "Classify customer support tickets into categories"
```

### 3. Add Input/Output Examples
```bash
# One at a time
prompt-opt add my_classifier --input "My order arrived broken" --output "product_damage"
prompt-opt add my_classifier --input "I was charged twice" --output "billing"

# Or bulk import from CSV
prompt-opt import my_classifier --file examples.csv

# Or from JSONL
prompt-opt import my_classifier --file examples.jsonl
```

### 4. Split the Dataset
```bash
prompt-opt split my_classifier
# → train: 140 examples, validation: 30, test: 30
```

### 5. Run Optimization
```bash
prompt-opt run my_classifier --max-rounds 30 --target-accuracy 0.95
```

The system will:
- Generate a baseline prompt automatically
- Run it against your train set
- Score every output
- Analyze failures, propose improvements
- Test the challenger on the validation set
- Promote if better, keep iterating
- Stop when target accuracy reached, budget exhausted, or plateaued

### 6. View Results
```bash
prompt-opt results my_classifier
prompt-opt report my_classifier --format markdown --output report.md
```

### 7. Export the Champion
```bash
prompt-opt export my_classifier --output best_config.yaml
```

The exported YAML is a self-contained config any developer can use directly:
```yaml
model: claude-sonnet-4-20250514
system_prompt: |
  You are a customer support ticket classifier. Classify each ticket
  into exactly one category: billing, product_damage, shipping, account, other.
  Respond with ONLY the category name. No explanation, no punctuation.
temperature: 0.0
max_tokens: 50
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.11+ | Ecosystem, async support |
| Package manager | uv | Fast, modern |
| CLI framework | Click | Explicit, composable, battle-tested |
| LLM provider | Anthropic (Claude) | v1 default, only provider |
| Validation | Pydantic v2 | Type-safe configs, serialization |
| Config files | YAML | Human-readable, good for prompts |
| Data files | JSONL | One record per line, easy to append |
| Terminal output | Rich | Tables, progress bars, colors |
| Testing | pytest + pytest-asyncio | Standard, async-compatible |
| Linting | Ruff | Fast, comprehensive |

---

## Tickets

These tickets should be built **in order** — each one builds on the previous:

| # | Ticket | What It Builds | Key Deliverable |
|---|--------|----------------|-----------------|
| 1 | [Project Scaffolding](ticket-1-project-scaffolding.md) | Package structure, config, CLI skeleton | `prompt-opt --help` works |
| 2 | [Dataset Store](ticket-2-dataset-store.md) | Input/output management, splitting | Users can add and manage eval examples |
| 3 | [Candidate Representation](ticket-3-candidate-representation.md) | Config schema, factory, diff | Candidates can be created, saved, compared |
| 4 | [Runner](ticket-4-runner.md) | LLM execution engine (Anthropic) | Candidates can be executed against examples |
| 5 | [Judge](ticket-5-judge.md) | Scoring engine (hard checks + LLM grader) | Outputs can be scored against expected |
| 6 | [Optimizer](ticket-6-optimizer.md) | Failure analysis + proposal engine | System can propose improvements |
| 7 | [Controller](ticket-7-controller.md) | Main loop orchestration | Full optimization loop runs end-to-end |
| 8 | [CLI & Reporting](ticket-8-cli-and-reporting.md) | User-facing commands, reports, export | Complete user experience |

---

## Non-Goals for v1

These are explicitly out of scope:

- **Ensembles / voting** — Deferred to v2
- **Routing between models per-input** — Deferred to v2
- **Multi-provider support (OpenAI, etc.)** — Anthropic only for v1
- **Fine-tuning integration** — Deferred to v3
- **Production deployment pipeline** — Deferred to v3
- **Web UI** — CLI only for now
- **RAG / retrieval as an optimization knob** — Deferred to v2
- **Automatic example generation** — Users provide all examples manually

---

## Success Criteria

- Given 50+ input/output examples for a well-defined task, the system finds a prompt+model config that scores within 5% of a human-crafted expert prompt within 20 optimization rounds
- The system plateaus gracefully (doesn't oscillate or regress)
- Cost of the optimization run is trackable and bounded
- The winning config is exportable as a standalone YAML that any developer can use directly
- The entire system can be installed and run with 4 commands: clone, sync, configure API key, run
