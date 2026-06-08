"""FastAPI app exposing TRELLIS.2 image-to-3D generation.

Endpoints
---------
GET  /health                         - liveness + which model variant is resident.
POST /generate                       - multipart/form-data: `image` file + `options` JSON.
POST /generate/base64                - application/json: { image_base64, options }.
GET  /results/{job_id}               - manifest for a `files`-mode result.
GET  /results/{job_id}/files/{name}  - download one raw (unzipped) file.

The `response_mode` option controls packaging: `auto` (raw single file, else zip),
`zip` (always zip), `files` (unzipped per-file download URLs), or `json` (base64
inline). See api/schemas.GenerateOptions, including the `quantization` flag that
selects native bf16 weights or the GGUF quantization.
"""
import base64
import io
import json
import zipfile
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.concurrency import run_in_threadpool
from PIL import Image

import torch

from . import config
from .generation import generate
from .pipeline_manager import manager
from .results_store import store
from .schemas import (
    Base64GenerateRequest,
    FileInfo,
    FilePayload,
    GenerateOptions,
    HealthResponse,
    JsonGenerateResponse,
    ResultManifest,
)

# (filename, media_type, bytes)
FileTuple = Tuple[str, str, bytes]

app = FastAPI(
    title="TRELLIS.2 Image-to-3D API",
    description=(
        "Generate textured 3D assets from a single image.\n\n"
        "**Input is image-only** — TRELLIS.2 has no text-to-3D pipeline.\n\n"
        "Outputs: GLB / OBJ / PLY / STL (+ optional turntable MP4). Use the "
        "`response_mode` option to get a single file, a zip, unzipped per-file "
        "download URLs (for testing), or base64-inline JSON."
    ),
    version="1.1.0",
    openapi_tags=[
        {"name": "generation", "description": "Submit an image, get a 3D asset back."},
        {"name": "results", "description": "Download individual files from a `files`-mode result."},
        {"name": "status", "description": "Health and readiness."},
    ],
)


@app.on_event("startup")
def _startup():
    if config.PRELOAD:
        try:
            manager.preload()
        except Exception as exc:  # don't crash the server; first request will retry
            print(f"[startup] preload failed ({exc}); will load on first request.")


@app.get("/health", response_model=HealthResponse, tags=["status"], summary="Liveness & loaded model")
def health():
    return HealthResponse(
        status="ok",
        cuda_available=torch.cuda.is_available(),
        device=manager.device,
        loaded_variant=manager.loaded_variant,
        default_variant=config.DEFAULT_VARIANT,
        allow_variant_switch=config.ALLOW_VARIANT_SWITCH,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
        try:
            return generate(pipeline, image, options)
        except HTTPException:
            raise
        except Exception as exc:  # one bad request must not take down the worker
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")


def _zip_bytes(files: List[FileTuple], metadata: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, _mt, data in files:
            zf.writestr(name, data)
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
    return buf.getvalue()


def _build_response(request: Request, metadata: dict, files: List[FileTuple], mode: str):
    """Package the generated files according to `response_mode`."""
    if mode == "binary":
        mode = "auto"
    meta_header = json.dumps(metadata)

    if mode == "json":
        payload = JsonGenerateResponse(
            metadata=metadata,
            files=[
                FilePayload(filename=n, media_type=mt, base64=base64.b64encode(d).decode())
                for n, mt, d in files
            ],
        )
        return JSONResponse(content=payload.model_dump())

    if mode == "files":
        job_id = store.put(metadata, files)
        base = str(request.base_url).rstrip("/")
        manifest = ResultManifest(
            job_id=job_id,
            metadata=metadata,
            files=[
                FileInfo(
                    filename=n, media_type=mt, size_bytes=len(d),
                    url=f"{base}/results/{job_id}/files/{n}",
                )
                for n, mt, d in files
            ],
            expires_in_sec=config.RESULTS_TTL_SEC,
        )
        return JSONResponse(content=manifest.model_dump())

    if mode == "zip" or len(files) != 1:
        return Response(
            content=_zip_bytes(files, metadata),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="trellis2_output.zip"',
                "X-Trellis-Metadata": meta_header,
            },
        )

    # auto + exactly one file -> raw
    name, media_type, data = files[0]
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "X-Trellis-Metadata": meta_header,
        },
    )


# --------------------------------------------------------------------------- #
# generation endpoints
# --------------------------------------------------------------------------- #
_GENERATE_RESPONSES = {
    200: {
        "description": "The generated asset. Shape depends on `response_mode`: raw "
                       "file (`model/gltf-binary`, ...), a `.zip`, a `files` manifest "
                       "(JSON), or base64-inline JSON.",
        "content": {
            "application/octet-stream": {},
            "application/zip": {},
            "model/gltf-binary": {},
            "application/json": {},
        },
    }
}


@app.post(
    "/generate",
    tags=["generation"],
    summary="Generate a 3D asset from an uploaded image",
    response_description="Asset bytes, a zip, a files manifest, or base64 JSON.",
    responses=_GENERATE_RESPONSES,
)
async def generate_multipart(
    request: Request,
    image: UploadFile = File(..., description="Input image (PNG/JPG/WebP). Alpha mask is used if present."),
    options: Optional[str] = Form(
        None,
        description='JSON-encoded GenerateOptions, e.g. '
                    '{"resolution":"512","formats":["glb","obj"],"response_mode":"files"}. '
                    "See the GenerateOptions schema (used directly by /generate/base64).",
    ),
):
    """Upload an image as multipart form data. `options` is a JSON string of
    GenerateOptions (resolution, formats, response_mode, quantization, ...)."""
    try:
        opts = GenerateOptions.model_validate_json(options) if options else GenerateOptions()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid options JSON: {exc}")
    img = _load_image(await image.read())
    metadata, files = await run_in_threadpool(_run_generation, img, opts)
    return _build_response(request, metadata, files, opts.response_mode)


@app.post(
    "/generate/base64",
    tags=["generation"],
    summary="Generate a 3D asset from a base64 image (pure JSON)",
    response_description="Asset bytes, a zip, a files manifest, or base64 JSON.",
    responses=_GENERATE_RESPONSES,
)
async def generate_base64(request: Request, req: Base64GenerateRequest):
    """JSON-only variant: handy for clients that prefer a single JSON body. The
    full GenerateOptions schema is documented inline here."""
    img = _decode_base64_image(req.image_base64)
    metadata, files = await run_in_threadpool(_run_generation, img, req.options)
    return _build_response(request, metadata, files, req.options.response_mode)


# --------------------------------------------------------------------------- #
# results endpoints (response_mode="files")
# --------------------------------------------------------------------------- #
@app.get(
    "/results/{job_id}",
    response_model=ResultManifest,
    tags=["results"],
    summary="Get the manifest (file list + URLs) for a files-mode result",
)
def get_result(request: Request, job_id: str):
    entry = store.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id.")
    base = str(request.base_url).rstrip("/")
    return ResultManifest(
        job_id=job_id,
        metadata=entry.metadata,
        files=[
            FileInfo(
                filename=name, media_type=mt, size_bytes=len(data),
                url=f"{base}/results/{job_id}/files/{name}",
            )
            for name, (mt, data) in entry.files.items()
        ],
        expires_in_sec=config.RESULTS_TTL_SEC,
    )


@app.get(
    "/results/{job_id}/files/{filename}",
    tags=["results"],
    summary="Download one raw (unzipped) file from a result",
    response_description="The raw file bytes with its native content type.",
    responses={200: {"content": {"application/octet-stream": {}}}, 404: {"description": "Not found / expired."}},
)
def get_result_file(job_id: str, filename: str):
    found = store.get_file(job_id, filename)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown job_id/filename or expired.")
    media_type, data = found
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
