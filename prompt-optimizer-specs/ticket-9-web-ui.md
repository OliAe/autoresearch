# Ticket 9: Web UI for Dataset Management

## Overview

A lightweight web interface for managing eval datasets — adding, editing, viewing, and importing input/output examples. The CLI is great for running optimizations, but manually typing `prompt-opt add --input "..." --output "..."` hundreds of times is impractical, especially for longer inputs/outputs.

The Web UI makes it easy for anyone to:
- Add examples through a form (with multi-line text areas)
- View all examples in a searchable, sortable table
- Edit or delete examples inline
- Bulk import via file upload (CSV/JSONL)
- Configure tasks and trigger dataset splits
- See dataset stats at a glance

**Tech stack:** FastAPI + Jinja2 templates + htmx (lightweight interactivity, no JS build step). Runs locally via `prompt-opt ui` and deploys to Railway in production (see Ticket 10).

---

## Acceptance Criteria

- [ ] `prompt-opt ui` starts a local web server (default: http://localhost:8000)
- [ ] `prompt-opt ui --port 3000` configurable port
- [ ] `prompt-opt ui --host 0.0.0.0` for Railway/production deployment
- [ ] Dashboard page: list all tasks with example counts and split status
- [ ] Task detail page: view all examples in a paginated table
- [ ] Add example page: form with text areas for input and output
- [ ] Edit example: inline editing in the table (htmx)
- [ ] Delete example: with confirmation
- [ ] Bulk import: file upload for CSV and JSONL
- [ ] Split controls: set ratios, trigger split, view split status
- [ ] Task creation: form to create a new task
- [ ] Task config: edit allowed models, scoring weights, description
- [ ] Search/filter examples by text content
- [ ] Pagination: 50 examples per page, with page navigation
- [ ] Responsive layout (works on laptop screens)
- [ ] Flash messages for success/error feedback
- [ ] API endpoints return JSON (for programmatic access alongside the HTML UI)
- [ ] Static assets (CSS) served by FastAPI (no CDN dependency for local use)
- [ ] Works behind Railway's reverse proxy (proper host/port handling)

---

## Tech Stack & Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
ui = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "jinja2>=3.1.0",
    "python-multipart>=0.0.9",   # for file uploads
]
```

Install with: `uv pip install -e ".[ui]"`

**Why these choices:**
- **FastAPI**: async-native, auto-generates OpenAPI docs, lightweight
- **Jinja2**: server-side rendering, no JS build step, fast to develop
- **htmx**: adds interactivity (inline edit, delete, search) without writing JavaScript — just HTML attributes
- **python-multipart**: required by FastAPI for file upload handling
- **No React/Vue/Next.js**: keeps the stack simple, no node_modules, no build step

---

## Routes & Pages

### Dashboard — `GET /`

The landing page. Shows all tasks.

```
╔══════════════════════════════════════════════════════════════╗
║  PROMPT OPTIMIZER                                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  Your Tasks                                    [+ New Task] ║
║                                                             ║
║  ┌────────────────┬──────────┬────────┬──────────────────┐  ║
║  │ Task           │ Examples │ Split? │ Actions          │  ║
║  ├────────────────┼──────────┼────────┼──────────────────┤  ║
║  │ my_classifier  │ 200      │ ✓      │ [View] [Config]  │  ║
║  │ email_summary  │ 50       │ ✗      │ [View] [Config]  │  ║
║  │ code_reviewer  │ 0        │ ✗      │ [View] [Config]  │  ║
║  └────────────────┴──────────┴────────┴──────────────────┘  ║
║                                                             ║
╚══════════════════════════════════════════════════════════════╝
```

**Route:** `GET /`
**Template:** `dashboard.html`
**Data:** List of all tasks with stats

---

### Task Detail — `GET /tasks/{task_name}`

View all examples for a task. The main workspace.

```
╔══════════════════════════════════════════════════════════════════════╗
║  ← Back    my_classifier                                           ║
║  Classify customer support tickets                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Stats: 200 examples │ Split: 140 train / 30 val / 30 test        ║
║                                                                    ║
║  [+ Add Example]  [Import CSV/JSONL]  [Split Dataset]  [Config]    ║
║                                                                    ║
║  Search: [________________________]                                ║
║                                                                    ║
║  ┌─────────┬──────────────────────────┬────────────────┬─────────┐ ║
║  │ ID      │ Input                    │ Output         │ Actions │ ║
║  ├─────────┼──────────────────────────┼────────────────┼─────────┤ ║
║  │ ex_001  │ My order arrived broken  │ product_damage │ [✎] [🗑]│ ║
║  │ ex_002  │ I was charged twice      │ billing        │ [✎] [🗑]│ ║
║  │ ex_003  │ When will my package...  │ shipping       │ [✎] [🗑]│ ║
║  │ ...     │ ...                      │ ...            │         │ ║
║  └─────────┴──────────────────────────┴────────────────┴─────────┘ ║
║                                                                    ║
║  Page: [< 1] [2] [3] [4] [>]                     Showing 1-50/200 ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Route:** `GET /tasks/{task_name}?page=1&search=`
**Template:** `task_detail.html`
**Features:**
- Paginated table (50 per page)
- Search filters examples by input or output text (case-insensitive substring)
- Input/output columns truncated to ~80 chars with full text on hover/click
- Edit button opens inline edit form (htmx `hx-get`)
- Delete button with confirmation dialog

---

### Add Example — `GET /tasks/{task_name}/add`

Form for adding a single example.

```
╔══════════════════════════════════════════════════════════════╗
║  ← Back to my_classifier                                   ║
║                                                             ║
║  Add Example                                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  ID (optional):  [________________]                         ║
║                  Leave blank to auto-generate                ║
║                                                             ║
║  Input:                                                     ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │                                                      │   ║
║  │  (multi-line text area, 6 rows tall)                 │   ║
║  │                                                      │   ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                             ║
║  Expected Output:                                           ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │                                                      │   ║
║  │  (multi-line text area, 4 rows tall)                 │   ║
║  │                                                      │   ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                             ║
║  Checks (optional):                                         ║
║  ☑ Exact match    ☐ Case sensitive                         ║
║  Must contain:   [________________]                         ║
║  Must not contain: [________________]                       ║
║                                                             ║
║  [Save & Add Another]   [Save & Go Back]                    ║
║                                                             ║
╚══════════════════════════════════════════════════════════════╝
```

**Routes:**
- `GET /tasks/{task_name}/add` — render form
- `POST /tasks/{task_name}/examples` — create example

**Key features:**
- Multi-line text areas (critical for pasting long inputs)
- "Save & Add Another" stays on the form (clears fields, shows success flash)
- "Save & Go Back" redirects to task detail page
- Client-side character count for input/output fields
- Optional check configuration (exact match, contains, etc.)

---

### Edit Example — `GET /tasks/{task_name}/examples/{example_id}/edit`

Either a separate page or an inline htmx partial.

**Routes:**
- `GET /tasks/{task_name}/examples/{example_id}/edit` — render edit form (full page or htmx partial)
- `PUT /tasks/{task_name}/examples/{example_id}` — update example
- `DELETE /tasks/{task_name}/examples/{example_id}` — delete example

**Inline editing with htmx:**
When the user clicks the edit button on a table row, htmx replaces that row with an editable form:

```html
<!-- Table row (read mode) -->
<tr id="example-ex_001">
  <td>ex_001</td>
  <td>My order arrived broken</td>
  <td>product_damage</td>
  <td>
    <button hx-get="/tasks/my_classifier/examples/ex_001/edit"
            hx-target="#example-ex_001"
            hx-swap="outerHTML">Edit</button>
    <button hx-delete="/tasks/my_classifier/examples/ex_001"
            hx-target="#example-ex_001"
            hx-swap="outerHTML"
            hx-confirm="Delete example ex_001?">Delete</button>
  </td>
</tr>

<!-- Table row (edit mode, returned by htmx) -->
<tr id="example-ex_001">
  <td>ex_001</td>
  <td><textarea name="input">My order arrived broken</textarea></td>
  <td><textarea name="expected_output">product_damage</textarea></td>
  <td>
    <button hx-put="/tasks/my_classifier/examples/ex_001"
            hx-target="#example-ex_001"
            hx-swap="outerHTML"
            hx-include="closest tr">Save</button>
    <button hx-get="/tasks/my_classifier/examples/ex_001/row"
            hx-target="#example-ex_001"
            hx-swap="outerHTML">Cancel</button>
  </td>
</tr>
```

---

### Import — `GET /tasks/{task_name}/import`

File upload form for bulk import.

```
╔══════════════════════════════════════════════════════════════╗
║  ← Back to my_classifier                                   ║
║                                                             ║
║  Import Examples                                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  Upload a CSV or JSONL file:                                ║
║                                                             ║
║  ┌──────────────────────────────────────────────┐           ║
║  │                                              │           ║
║  │   Drag & drop file here, or click to browse  │           ║
║  │                                              │           ║
║  │   Supported: .csv, .jsonl                    │           ║
║  │                                              │           ║
║  └──────────────────────────────────────────────┘           ║
║                                                             ║
║  CSV Column Mapping:                                        ║
║  Input column:  [input________▼]                            ║
║  Output column: [output_______▼]                            ║
║  ID column:     [auto-generate▼]                            ║
║                                                             ║
║  [Upload & Import]                                          ║
║                                                             ║
║  ─────────────────────────────────────                      ║
║  CSV format:                                                ║
║  input,output                                               ║
║  "My order is late","shipping"                              ║
║  "Charged twice","billing"                                  ║
║                                                             ║
║  JSONL format:                                              ║
║  {"input": "My order is late", "output": "shipping"}        ║
║  {"input": "Charged twice", "output": "billing"}            ║
║                                                             ║
╚══════════════════════════════════════════════════════════════╝
```

**Routes:**
- `GET /tasks/{task_name}/import` — render upload form
- `POST /tasks/{task_name}/import` — handle file upload and import

**Behavior:**
1. User uploads file
2. Server detects format, validates columns
3. Imports examples
4. Redirects to task detail with flash: "Imported 150 examples (3 skipped)"

---

### Split — `POST /tasks/{task_name}/split`

Split controls appear on the task detail page (not a separate page).

```html
<!-- Split section on task detail page -->
<div id="split-controls">
  <h3>Dataset Split</h3>
  <form hx-post="/tasks/my_classifier/split" hx-target="#split-status">
    <label>Train: <input type="number" name="train" value="0.7" step="0.05" min="0.1" max="0.9"></label>
    <label>Val: <input type="number" name="val" value="0.15" step="0.05" min="0.05" max="0.5"></label>
    <label>Test: <input type="number" name="test" value="0.15" step="0.05" min="0.05" max="0.5"></label>
    <label>Seed: <input type="number" name="seed" value="42"></label>
    <button type="submit">Split Dataset</button>
  </form>
  <div id="split-status">
    <!-- Updated via htmx after split -->
    <p>Current split: 140 train / 30 val / 30 test (seed: 42)</p>
  </div>
</div>
```

---

### Task Config — `GET /tasks/{task_name}/config`

Edit task configuration.

```
╔══════════════════════════════════════════════════════════════╗
║  ← Back to my_classifier                                   ║
║                                                             ║
║  Task Configuration                                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  Name:        my_classifier                                 ║
║  Description: [Classify customer support tickets_______]    ║
║                                                             ║
║  Allowed Models:                                            ║
║  ☑ claude-sonnet-4-20250514                                ║
║  ☐ claude-haiku-4-5-20251001                               ║
║  ☐ claude-opus-4-20250514                                  ║
║                                                             ║
║  Scoring:                                                   ║
║  Hard check weight:  [0.7___]                               ║
║  LLM grade weight:   [0.3___]                               ║
║  Pass threshold:     [0.8___]                               ║
║  Grading model:      [claude-sonnet-4-20250514▼]           ║
║                                                             ║
║  [Save Configuration]                                       ║
║                                                             ║
╚══════════════════════════════════════════════════════════════╝
```

**Routes:**
- `GET /tasks/{task_name}/config` — render config form
- `PUT /tasks/{task_name}/config` — update config

---

### Create Task — `GET /tasks/new`

```
╔══════════════════════════════════════════════════════════════╗
║  Create New Task                                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                             ║
║  Task Name:    [________________]                           ║
║               (alphanumeric + underscores)                  ║
║                                                             ║
║  Description:  [________________________________]           ║
║                                                             ║
║  Allowed Models:                                            ║
║  ☑ claude-sonnet-4-20250514                                ║
║  ☐ claude-haiku-4-5-20251001                               ║
║  ☐ claude-opus-4-20250514                                  ║
║                                                             ║
║  [Create Task]                                              ║
║                                                             ║
╚══════════════════════════════════════════════════════════════╝
```

**Routes:**
- `GET /tasks/new` — render form
- `POST /tasks` — create task, redirect to task detail

---

## API Endpoints (JSON)

Every route also serves JSON when requested with `Accept: application/json` header or `.json` suffix. This allows programmatic access.

```
GET    /api/tasks                                → list all tasks
POST   /api/tasks                                → create task
GET    /api/tasks/{name}                         → task details + stats
PUT    /api/tasks/{name}/config                  → update config
GET    /api/tasks/{name}/examples?page=1&search= → paginated examples
POST   /api/tasks/{name}/examples                → add example
PUT    /api/tasks/{name}/examples/{id}           → update example
DELETE /api/tasks/{name}/examples/{id}           → delete example
POST   /api/tasks/{name}/import                  → bulk import (file upload)
POST   /api/tasks/{name}/split                   → trigger split
GET    /api/tasks/{name}/stats                   → dataset statistics
```

---

## FastAPI Application Structure

```python
# src/prompt_optimizer/web/__init__.py
# src/prompt_optimizer/web/app.py         — FastAPI app factory
# src/prompt_optimizer/web/routes.py      — all route handlers
# src/prompt_optimizer/web/templates/     — Jinja2 templates
# src/prompt_optimizer/web/static/        — CSS, htmx.min.js

```

### File Layout

```
src/prompt_optimizer/web/
├── __init__.py
├── app.py                      # FastAPI app factory, lifespan, middleware
├── routes.py                   # All route handlers (HTML + JSON)
├── dependencies.py             # Dependency injection (Settings, Dataset)
├── templates/
│   ├── base.html               # Base layout (nav, flash messages, htmx script)
│   ├── dashboard.html          # Task list
│   ├── task_detail.html        # Example table + split controls
│   ├── task_new.html           # Create task form
│   ├── task_config.html        # Edit task config
│   ├── example_add.html        # Add example form
│   ├── example_import.html     # File upload form
│   └── partials/
│       ├── example_row.html    # Single table row (for htmx swaps)
│       ├── example_edit.html   # Inline edit form (htmx partial)
│       ├── split_status.html   # Split status panel (htmx partial)
│       └── flash.html          # Flash message partial
└── static/
    ├── style.css               # Minimal CSS (clean, modern, responsive)
    └── htmx.min.js             # htmx library (vendored, no CDN)
```

### App Factory

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Can be called with custom settings for testing,
    or with defaults for production.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="Prompt Optimizer",
        description="Eval-driven LLM program optimization",
    )

    # Static files
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Templates
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=templates_dir)

    # Store in app state
    app.state.settings = settings
    app.state.templates = templates

    # Register routes
    from .routes import router
    app.include_router(router)

    return app
```

### CLI Entry Point

```python
# In cli.py, add to the Click group:

@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev mode)")
@click.pass_context
def ui(ctx, host, port, reload):
    """Start the web UI for managing datasets."""
    import uvicorn
    from prompt_optimizer.web.app import create_app

    settings = ctx.obj["settings"]
    app = create_app(settings)

    console.print(f"Starting Prompt Optimizer UI at http://{host}:{port}")
    uvicorn.run(
        "prompt_optimizer.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
```

For Railway, the UI is started via:
```bash
uvicorn prompt_optimizer.web.app:create_app --host 0.0.0.0 --port $PORT --factory
```

---

## CSS / Styling

Use minimal, clean CSS — no heavy framework. The UI should feel modern but be fast and functional.

**Design principles:**
- Clean sans-serif font (system font stack)
- White background, subtle borders
- Tables with alternating row colors
- Form fields with clear labels and generous padding
- Success = green, Error = red, Warning = amber
- Responsive: works down to ~900px width (laptop)
- No animations except subtle fade for htmx swaps

**Base layout (base.html):**
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}Prompt Optimizer{% endblock %}</title>
    <link rel="stylesheet" href="/static/style.css">
    <script src="/static/htmx.min.js"></script>
</head>
<body>
    <nav>
        <a href="/" class="logo">Prompt Optimizer</a>
    </nav>

    {% if flash_messages %}
    <div class="flash-container">
        {% for msg in flash_messages %}
        <div class="flash flash-{{ msg.type }}">{{ msg.text }}</div>
        {% endfor %}
    </div>
    {% endif %}

    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

**htmx is vendored** (included as a static file, not loaded from CDN). This ensures the UI works on local networks and in Railway without external dependencies.

---

## Railway Considerations

The Web UI must work when deployed to Railway:

1. **Port binding:** Railway sets `$PORT` env var — the app must bind to `0.0.0.0:$PORT`
2. **No local filesystem persistence:** Railway's filesystem is ephemeral. Datasets must be stored in a **persistent volume** or a database. For v1, use Railway's **volume mounts** to persist the `datasets/` and `runs/` directories.
3. **Environment variables:** `ANTHROPIC_API_KEY` and other config set via Railway's env var UI
4. **Health check:** Add a `GET /health` endpoint that returns `{"status": "ok"}` for Railway's health checks
5. **CORS:** Not needed for v1 (server-rendered HTML, no separate frontend)
6. **HTTPS:** Railway provides this automatically via their proxy — the app serves HTTP internally

```python
# Health check endpoint
@router.get("/health")
async def health():
    return {"status": "ok"}
```

---

## Testing Requirements

```
# App creation
test_create_app_returns_fastapi
test_create_app_with_custom_settings

# Dashboard
test_dashboard_lists_tasks
test_dashboard_empty_state

# Task detail
test_task_detail_shows_examples
test_task_detail_pagination
test_task_detail_search_filter
test_task_detail_task_not_found_404

# Add example
test_add_example_form_renders
test_add_example_post_success
test_add_example_post_empty_input_error
test_add_example_post_empty_output_error
test_add_example_save_and_add_another

# Edit example
test_edit_example_htmx_partial
test_edit_example_put_success
test_edit_example_not_found

# Delete example
test_delete_example_success
test_delete_example_not_found

# Import
test_import_form_renders
test_import_csv_success
test_import_jsonl_success
test_import_invalid_format_error

# Split
test_split_post_success
test_split_invalid_ratios_error

# Task creation
test_create_task_form_renders
test_create_task_post_success
test_create_task_duplicate_name_error

# Task config
test_config_form_renders
test_config_update_success

# API endpoints (JSON)
test_api_list_tasks
test_api_get_task
test_api_create_example
test_api_update_example
test_api_delete_example
test_api_import

# Health check
test_health_endpoint

# Railway
test_app_respects_port_env_var
test_app_binds_to_0_0_0_0_in_production
```

Use FastAPI's `TestClient` for all tests.

**Minimum: 30 test cases.**

---

## Implementation Notes

- The web app reuses the `Dataset` class from Ticket 2 — no data access duplication
- htmx is the only client-side JS — no build step, no npm, no bundler
- Vendor htmx.min.js (download and commit it to static/) so it works offline and on Railway
- Templates use Jinja2's template inheritance (`base.html` → page templates)
- Flash messages stored in session (use FastAPI's `SessionMiddleware` with a secret key from Settings)
- For file uploads, use `UploadFile` from FastAPI — files are saved to a temp dir, then imported
- The web app shares the same `Settings` and `Dataset` classes as the CLI
- For Railway, the `datasets/` directory must be on a persistent volume mount

---

## Dependencies

- **Depends on:** Ticket 1 (Settings, config), Ticket 2 (Dataset, Example, TaskConfig — all data access)
- **Depended on by:** Ticket 10 (Railway deployment serves this app)
