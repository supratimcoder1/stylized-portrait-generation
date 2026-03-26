# FastAPI + Jinja2 Deployment Plan

## Goal
Serve the trained stylization model through a web UI and API that are stable, testable, and production-ready.

## Phase 1: Codebase Stabilization

- Consolidate inference logic in one shared module.
- Keep backward-compatible CLI interface for local workflows.
- Enforce typed stylization parameters and clamped runtime values.

Status: Completed in current refactor.

## Phase 2: API and Frontend Foundation

- Expose `/health` endpoint for readiness checks.
- Expose `/api/stylize` endpoint with image upload + style controls.
- Render a Jinja2 homepage with upload form and output preview.
- Serve generated outputs and static assets from known folders.

Status: Completed in current refactor.

## Phase 3: Validation and Safety

- Enforce upload size limits and MIME validation.
- Add user-facing error messages for invalid files.
- Add request-level timeouts and circuit breakers for model failures.
- Add generated-file cleanup policy (TTL or capped disk usage).

Status: Pending.

## Phase 4: Testing

- Unit tests for preprocess/postprocess and style slider behavior.
- API tests for `/health` and `/api/stylize` success and failure paths.
- Startup test to ensure checkpoint file loads in deployment environment.

Status: Pending.

## Phase 5: Productionization

- Add container image for repeatable deployments.
- Configure Gunicorn/Uvicorn worker strategy for CPU inference.
- Add structured logs with request IDs and latency metrics.
- Add CI pipeline for lint, tests, and build validation.

Status: Pending.

## Suggested Runtime Commands

- Local dev: `uvicorn backend.main:app --reload`
- Production baseline: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

## Risks to Address Before Production

- Large image uploads causing memory pressure.
- Concurrent requests saturating CPU and degrading latency.
- Unbounded accumulation in generated output directory.
- Missing model file in deployment artifact.
