# Stylized Portrait Generation

This repository trains and serves a U-Net based portrait stylization model.

## Clean Frontend/Backend Layout

- `frontend/`: Jinja2 templates and static assets
	- `frontend/templates/`
	- `frontend/static/`
- `backend/`: backend application and model runtime code
	- `backend/portrait_gen/api/`
	- `backend/portrait_gen/inference/`
	- `backend/portrait_gen/training/`
	- `backend/portrait_gen/cli/`
	- `backend/tools/data_prep/`: dataset prep and utility scripts
- `assets/`: non-runtime project artifacts
	- `assets/archives/`: zipped datasets/models
	- `assets/examples/`: sample input/output images
- `generated/`: runtime stylization outputs written by the API

Compatibility wrappers remain in `app/`, `training/`, `pipeline/`, and `scripts/` so old commands still work.

## Setup

```bash
pip install -r requirements.txt
```

## Run CLI Inference

```bash
python -m backend.portrait_gen.cli.infer \
	--input path/to/input.jpg \
	--output path/to/output.png \
	--weights checkpoints/final_model.pth
```

Legacy command still works:

```bash
python pipeline/complete_pipeline.py --input path/to/input.jpg --output path/to/output.png
```

## Run Web App (FastAPI + Jinja2)

```bash
uvicorn backend.main:app --reload
```

Legacy app module still works:

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Deployment Plan (Execution Checklist)

1. Finalize API contract and input validation rules (max upload size, allowed formats, slider ranges).
2. Add structured logging and request IDs for inference observability.
3. Add tests:
	 - unit tests for preprocessing/postprocessing
	 - API tests for upload and health endpoints
	 - smoke test for model load at startup
4. Containerize service with CPU-friendly runtime image and explicit startup command.
5. Add production settings (`workers`, timeouts, max request body, temp storage policy).
6. Add cache/retention policy for `generated/` outputs.
7. Add CI jobs for lint + tests + image build.
8. Deploy to staging first, benchmark latency, then promote to production.