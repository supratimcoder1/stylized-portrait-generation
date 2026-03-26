from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.portrait_gen.api.config import CHECKPOINT_PATH, GENERATED_DIR, STATIC_DIR, TEMPLATES_DIR
from backend.portrait_gen.inference.service import StylizeParams, VectorPortraitService

app = FastAPI(title="Stylized Portrait Generator")

if not CHECKPOINT_PATH.exists():
    raise RuntimeError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

GENERATED_DIR.mkdir(parents=True, exist_ok=True)

service = VectorPortraitService(weights_path=CHECKPOINT_PATH, device="cpu")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model": CHECKPOINT_PATH.name}


@app.post("/api/stylize")
async def stylize_image(
    image: UploadFile = File(...),
    num_colors: int = Form(8),
    edge_weight: float = Form(0.0),
    saturation: float = Form(1.0),
    contrast: float = Form(1.0),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Missing filename in uploaded file")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    params = StylizeParams(
        num_colors=max(2, min(32, num_colors)),
        edge_weight=max(0.0, min(1.0, edge_weight)),
        saturation=max(0.5, min(2.0, saturation)),
        contrast=max(0.5, min(2.0, contrast)),
    )

    try:
        output = service.generate_from_array(decoded, restore_size=True, params=params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stylization failed: {exc}") from exc

    output_name = f"{uuid4().hex}.png"
    output_path = GENERATED_DIR / output_name

    if not cv2.imwrite(str(output_path), output):
        raise HTTPException(status_code=500, detail="Failed to save generated output")

    return JSONResponse(
        {
            "output_url": f"/generated/{output_name}",
            "output_file": output_name,
        }
    )
