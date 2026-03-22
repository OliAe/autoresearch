# Ticket 10: Railway Deployment

## Overview

Deploy the Prompt Optimizer to Railway so it can run as a hosted service. This covers the Dockerfile, Railway configuration, persistent storage for datasets and runs, environment variable management, health checks, and the production startup command.

Railway is a PaaS that deploys from a Git repo or Docker image. We'll use a **Dockerfile** for maximum control over the build, with Railway's **volume mounts** for persistent data storage.

---

## Acceptance Criteria

- [ ] `Dockerfile` builds a production-ready image
- [ ] `railway.toml` or `railway.json` configures the Railway deployment
- [ ] App binds to `0.0.0.0:$PORT` (Railway injects `$PORT`)
- [ ] `GET /health` returns 200 for Railway health checks
- [ ] `datasets/` and `runs/` directories persist across deploys via Railway volumes
- [ ] `ANTHROPIC_API_KEY` set via Railway environment variables (not in code/image)
- [ ] App starts correctly with a single command
- [ ] Docker image is reasonably sized (< 500MB)
- [ ] `.dockerignore` excludes unnecessary files
- [ ] The CLI (`prompt-opt`) is available inside the container for manual operations
- [ ] Logs go to stdout/stderr (Railway captures these automatically)
- [ ] Graceful shutdown on SIGTERM (Railway sends this on redeploy)
- [ ] Works with Railway's automatic HTTPS (app serves HTTP internally)
- [ ] Documentation for setting up the Railway project from scratch

---

## Dockerfile

```dockerfile
# ──────────────────────────────────────────────
# Stage 1: Build
# ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps in production)
RUN uv sync --frozen --no-dev --extra ui

# Copy source code
COPY src/ src/
COPY README.md ./

# Install the package itself
RUN uv pip install --no-deps -e .

# ──────────────────────────────────────────────
# Stage 2: Runtime
# ──────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages and source from builder
COPY --from=builder /app /app

# Create directories for persistent data
# These will be mounted as Railway volumes in production
RUN mkdir -p /data/datasets /data/runs

# Set environment defaults
# Railway will override PORT; ANTHROPIC_API_KEY set via Railway env vars
ENV PROMPT_OPT_DATASETS_DIR=/data/datasets \
    PROMPT_OPT_RUNS_DIR=/data/runs \
    PYTHONUNBUFFERED=1

# Expose port (documentation only — Railway uses $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

# Start the web UI
CMD ["sh", "-c", "uv run uvicorn prompt_optimizer.web.app:create_app --host 0.0.0.0 --port ${PORT:-8000} --factory --workers 1"]
```

### Why these choices:
- **Multi-stage build:** Keeps the runtime image smaller (no build tools)
- **`uv sync --frozen`:** Reproducible installs from lockfile
- **`--no-dev --extra ui`:** Only production + UI dependencies (no pytest, ruff)
- **`/data/` directory:** Clean separation from app code — mounted as Railway volume
- **`PYTHONUNBUFFERED=1`:** Ensures logs appear immediately in Railway's log viewer
- **Single worker:** v1 is single-user, no need for multiprocessing
- **`sh -c`:** Allows `$PORT` env var expansion at runtime

---

## .dockerignore

```
# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
.pytest_cache
.ruff_cache
*.egg-info

# Virtual envs
.venv
venv

# IDE
.vscode
.idea

# Local data (will be on Railway volumes)
datasets/
runs/

# Env files (secrets set via Railway)
.env
.env.*

# Docs/specs (not needed in production)
prompt-optimizer-specs/
EPIC-prompt-optimizer.md
*.md
!README.md

# Large files
*.png
*.jpg
progress.png
analysis.ipynb
```

---

## Railway Configuration

### `railway.toml`

```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 5
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

### Alternative: `railway.json`

```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 5,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

---

## Environment Variables (Railway)

Set these in Railway's dashboard under the service's "Variables" tab:

| Variable | Required | Example | Notes |
|----------|----------|---------|-------|
| `ANTHROPIC_API_KEY` | YES | `sk-ant-api03-...` | Your Anthropic API key |
| `PORT` | AUTO | `8000` | Railway injects this automatically |
| `PROMPT_OPT_DATASETS_DIR` | NO | `/data/datasets` | Default in Dockerfile |
| `PROMPT_OPT_RUNS_DIR` | NO | `/data/runs` | Default in Dockerfile |
| `PROMPT_OPT_DEFAULT_MODEL` | NO | `claude-sonnet-4-20250514` | Override default model |
| `PROMPT_OPT_MAX_CONCURRENT_REQUESTS` | NO | `10` | API concurrency limit |
| `SECRET_KEY` | YES | `random-string-here` | For session middleware (flash messages) |

**NEVER put `ANTHROPIC_API_KEY` in the Dockerfile, code, or git.**

---

## Persistent Storage (Railway Volumes)

Railway's filesystem is **ephemeral** — it resets on every deploy. Without volumes, all datasets and run history would be lost.

### Setup:
1. In Railway dashboard, go to your service
2. Click "Volumes"
3. Add a volume:
   - **Mount path:** `/data`
   - **Size:** 1 GB (adjust as needed)

This persists the entire `/data` directory (which contains `datasets/` and `runs/`) across deploys and restarts.

### Volume structure:
```
/data/                    # Railway persistent volume
├── datasets/             # All task datasets
│   ├── my_classifier/
│   │   ├── config.yaml
│   │   ├── examples.jsonl
│   │   └── splits/
│   └── email_summarizer/
│       └── ...
└── runs/                 # All optimization runs
    ├── run_20260322_143000/
    │   ├── champion.yaml
    │   ├── report.md
    │   └── rounds/
    └── ...
```

---

## Settings Updates

The `Settings` class (from Ticket 1) needs to support Railway's environment:

```python
class Settings(BaseSettings):
    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-20250514"
    default_temperature: float = 0.0
    max_concurrent_requests: int = 10
    datasets_dir: Path = Path("datasets")          # overridden by PROMPT_OPT_DATASETS_DIR
    runs_dir: Path = Path("runs")                  # overridden by PROMPT_OPT_RUNS_DIR
    secret_key: str = "dev-secret-change-me"       # for session middleware
    environment: str = "development"               # "development" or "production"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PROMPT_OPT_",
        extra="ignore",                            # ignore unknown env vars (Railway sets many)
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"
```

In Railway, set `PROMPT_OPT_ENVIRONMENT=production` to enable production behavior (e.g., disable debug mode, enforce secret key).

---

## Production Startup

The app starts with a single command (defined in Dockerfile CMD):

```bash
uvicorn prompt_optimizer.web.app:create_app \
    --host 0.0.0.0 \
    --port $PORT \
    --factory \
    --workers 1 \
    --log-level info
```

**Why `--workers 1`:**
- v1 is single-user/small-team
- The `Dataset` class uses file-based storage (JSONL) — concurrent writes from multiple workers would corrupt data
- If scaling is needed later, move to a database and increase workers

**Graceful shutdown:**
- Uvicorn handles SIGTERM gracefully by default
- Railway sends SIGTERM on redeploy, waits 10s, then SIGKILL
- In-progress API calls (optimization runs) will be interrupted — the Controller's round-level persistence (Ticket 7) ensures no data loss

---

## Running the Optimization on Railway

The web UI handles dataset management. To run optimizations, you have two options:

### Option A: Via the Web UI (future ticket)
Add a "Run Optimization" button to the web UI that triggers the Controller. This is out of scope for v1 but is the natural next step.

### Option B: Via Railway's CLI / shell
```bash
# SSH into the Railway container
railway shell

# Run optimization from inside the container
uv run prompt-opt run my_classifier --max-rounds 30 --target-accuracy 0.95
```

### Option C: Add an API endpoint (recommended for v1)
Add a simple API endpoint that triggers optimization:

```python
@router.post("/api/tasks/{task_name}/optimize")
async def start_optimization(
    task_name: str,
    max_rounds: int = 50,
    target_accuracy: float = 0.95,
    max_budget: float = 50.0,
    background_tasks: BackgroundTasks,
):
    """Start an optimization run in the background."""
    settings = get_settings()
    controller = Controller(
        task_name=task_name,
        settings=settings,
        max_rounds=max_rounds,
        target_accuracy=target_accuracy,
        max_budget_usd=max_budget,
    )
    background_tasks.add_task(controller.run)
    return {"status": "started", "message": f"Optimization started for {task_name}"}


@router.get("/api/tasks/{task_name}/optimize/status")
async def optimization_status(task_name: str):
    """Check status of running optimization."""
    # Read latest run directory for this task
    ...
```

This lets you trigger optimization from the web UI or via API call, without needing SSH access to Railway.

---

## Deployment Walkthrough

Step-by-step guide for deploying to Railway from scratch:

### 1. Prerequisites
- Railway account (https://railway.app)
- Railway CLI installed: `npm install -g @railway/cli`
- Git repository with all code

### 2. Create Railway Project
```bash
# Login to Railway
railway login

# Initialize project
railway init

# Link to your repo (or Railway will auto-detect from GitHub)
railway link
```

### 3. Add Persistent Volume
- Go to Railway dashboard → your service → "Volumes"
- Add volume with mount path `/data`

### 4. Set Environment Variables
In Railway dashboard → service → "Variables":
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
PROMPT_OPT_ENVIRONMENT=production
SECRET_KEY=generate-a-random-string-here
```

### 5. Deploy
```bash
# Push to deploy
railway up

# Or connect GitHub repo for auto-deploy on push
```

### 6. Verify
- Open the Railway-provided URL
- Check `/health` returns `{"status": "ok"}`
- Create a task, add examples, verify persistence across redeployments

### 7. Custom Domain (optional)
In Railway dashboard → service → "Settings" → "Networking" → Add custom domain.

---

## Testing Requirements

```
# Dockerfile
test_dockerfile_builds_successfully
test_docker_image_starts
test_docker_image_health_check
test_docker_image_respects_port_env
test_docker_image_cli_available  # prompt-opt --help works inside container

# Settings
test_settings_reads_railway_env_vars
test_settings_datasets_dir_override
test_settings_runs_dir_override
test_settings_ignores_unknown_env_vars
test_settings_is_production_flag

# Health endpoint
test_health_returns_200
test_health_returns_json

# Volume persistence
test_data_persists_in_volume_dir
test_datasets_created_in_configured_dir
test_runs_created_in_configured_dir

# Production startup
test_uvicorn_starts_with_factory_flag
test_app_binds_to_0_0_0_0

# Optimize API endpoint
test_optimize_endpoint_starts_background_task
test_optimize_status_endpoint
test_optimize_task_not_found_404
```

Use Docker SDK for Python (`docker` package) for container tests.
Use FastAPI's `TestClient` for endpoint tests.

**Minimum: 15 test cases.**

---

## Implementation Notes

- The Dockerfile should be at the repo root (Railway expects this by default)
- Use `.dockerignore` aggressively — keep the image small
- Railway auto-detects Dockerfiles — no additional build configuration needed
- For local development, continue using `prompt-opt ui` (no Docker needed)
- For production testing, use `docker build . -t prompt-opt && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... prompt-opt`
- Railway provides automatic HTTPS — the app should NOT configure TLS/SSL itself
- Railway's log viewer captures stdout/stderr — use Python's `logging` module, not print()
- Consider adding a `Procfile` as a fallback: `web: uvicorn prompt_optimizer.web.app:create_app --host 0.0.0.0 --port $PORT --factory`

---

## Dependencies

- **Depends on:** Ticket 1 (Settings), Ticket 9 (Web UI — this is what gets deployed)
- **This is the final infrastructure ticket.**

---

## Future Enhancements (not in v1)

- **Auto-deploy on push:** Connect Railway to GitHub for CI/CD
- **Preview environments:** Railway can create ephemeral environments per PR
- **Database migration:** If file-based storage hits limits, migrate to PostgreSQL (Railway has a Postgres plugin)
- **Worker separation:** Split the web UI and optimization runner into separate Railway services (web + worker)
- **Monitoring:** Add Railway's built-in metrics or integrate with an external service
