"""FastAPI app exposing TRELLIS.2 image-to-3D generation.

Endpoints
---------
GET  /health           - liveness + which model variant is resident.
POST /generate         - multipart/form-data: `image` file + optional `options` JSON.
POST /generate/base64  - application/json: { image_base64, options }.

Both generate endpoints share the same options (see api/schemas.GenerateOptions),
including the `quantization` flag that selects native bf16 weights or the GGUF
quantization ("gguf-q8" / "gguf-q4" / "gguf" + `gguf_quant`).
"""
import base64
import io
import json
import zipfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.concurrency import run_in_threadpool
from PIL import Image

import torch

from . import config
from .generation import generate
from .pipeline_manager import manager
from .schemas import (
    Base64GenerateRequest,
    GenerateOptions,
    HealthResponse,
    JsonGenerateResponse,
    FilePayload,
)

app = FastAPI(
    title="TRELLIS.2 Image-to-3D API",
    description="Generate textured 3D assets from a single image. Image input only "
                "(TRELLIS.2 does not support text-to-3D).",
    version="1.0.0",
)


@app.on_event("startup")
def _startup():
    if config.PRELOAD:
        try:
            manager.preload()
        except Exception as exc:  # don't crash the server; first request will retry
            print(f"[startup] preload failed ({exc}); will load on first request.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        cuda_available=torch.cuda.is_available(),
        device=manager.device,
        loaded_variant=manager.loaded_variant,
        default_variant=config.DEFAULT_VARIANT,
        allow_variant_switch=config.ALLOW_VARIANT_SWITCH,
    )


def _load_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")


def _decode_base64_image(s: str) -> Image.Image:
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    try:
        return _load_image(base64.b64decode(s))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")


def _run_generation(image: Image.Image, options: GenerateOptions):
    """Resolve the variant, serialize on the GPU, and generate."""
    try:
        pipeline = manager.get(options.quantization, options.gguf_quant)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    pipeline._active_variant = manager.loaded_variant
    with manager.run_lock:
        return generate(pipeline, image, options)


def _build_response(metadata: dict, files, response_mode: str):
    """Binary (single file or zip) or JSON (base64) response."""
    meta_header = json.dumps(metadata)

    if response_mode == "json":
        payload = JsonGenerateResponse(
            metadata=metadata,
            files=[
                FilePayload(filename=name, media_type=mt, base64=base64.b64encode(data).decode())
                for name, (mt, data) in files.items()
            ],
        )
        return JSONResponse(content=payload.model_dump())

    if len(files) == 1:
        name, (media_type, data) = next(iter(files.items()))
        return Response(
            content=data,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{name}"',
                "X-Trellis-Metadata": meta_header,
            },
        )

    # Multiple files -> zip them together.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, (_mt, data) in files.items():
            zf.writestr(name, data)
        zf.writestr("metadata.json", meta_header)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="trellis2_output.zip"',
            "X-Trellis-Metadata": meta_header,
        },
    )


@app.post("/generate")
async def generate_multipart(
    image: UploadFile = File(..., description="Input image (PNG/JPG/WebP). Alpha mask is used if present."),
    options: Optional[str] = Form(None, description="JSON-encoded GenerateOptions."),
):
    opts = GenerateOptions.model_validate_json(options) if options else GenerateOptions()
    img = _load_image(await image.read())
    metadata, files = await run_in_threadpool(_run_generation, img, opts)
    return _build_response(metadata, files, opts.response_mode)


@app.post("/generate/base64")
async def generate_base64(req: Base64GenerateRequest):
    img = _decode_base64_image(req.image_base64)
    metadata, files = await run_in_threadpool(_run_generation, img, req.options)
    return _build_response(metadata, files, req.options.response_mode)
