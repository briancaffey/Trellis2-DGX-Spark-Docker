# TRELLIS.2 FastAPI Service

A small HTTP API around the native `Trellis2ImageTo3DPipeline`. Send an image,
get back a textured 3D asset in one or more formats.

> **Input is image-only.** TRELLIS.2 does **not** support text-to-3D — there is no
> text-conditioned pipeline in the model. Give it a single image (an alpha-masked
> foreground works best; otherwise the background is removed automatically).

## Run it

```bash
# build + start the API service (separate from the Gradio app)
docker compose up --build api          # service name: trellis2-api

# served on http://localhost:8000  (override with API_PORT in .env)
curl http://localhost:8000/health
```

Interactive docs (Swagger UI) are at `http://localhost:8000/docs`.

## Endpoints

| Method | Path               | Body                                                        |
|--------|--------------------|-------------------------------------------------------------|
| GET    | `/health`          | —                                                           |
| POST   | `/generate`        | `multipart/form-data`: `image` file + optional `options` JSON |
| POST   | `/generate/base64` | `application/json`: `{ "image_base64": "...", "options": {…} }` |

### Quick examples

Single GLB back, default settings:

```bash
curl -X POST http://localhost:8000/generate \
  -F image=@chair.png \
  -o chair.glb
```

Multiple formats + options (returns a ZIP because >1 file is requested):

```bash
curl -X POST http://localhost:8000/generate \
  -F image=@chair.png \
  -F 'options={"resolution":"1024","formats":["glb","ply"],"texture_size":2048}' \
  -o chair.zip
```

JSON in / JSON out (base64-encoded files + metadata), handy for code clients:

```bash
curl -X POST http://localhost:8000/generate/base64 \
  -H 'Content-Type: application/json' \
  -d '{"image_base64":"data:image/png;base64,iVBOR...","options":{"response_mode":"json"}}'
```

## `options` reference

| Field               | Type / values                                  | Default     | Notes |
|---------------------|------------------------------------------------|-------------|-------|
| `seed`              | int                                            | `42`        | |
| `randomize_seed`    | bool                                           | `false`     | Ignores `seed` when true |
| `resolution`        | `"512"` \| `"1024"` \| `"1536"`                | `"1024"`    | Higher = sharper, slower, more VRAM |
| `formats`           | list of `glb,obj,ply,stl,preview`              | `["glb"]`   | >1 ⇒ ZIP response |
| `response_mode`     | `"binary"` \| `"json"`                         | `"binary"`  | `json` returns base64 + metadata |
| `decimation_target` | int                                            | `500000`    | Target face count for the exported mesh |
| `texture_size`      | int                                            | `2048`      | Baked texture resolution |
| `quantization`      | `"bf16"` \| `"gguf-q8"` \| `"gguf-q4"` \| `"gguf"` | server default | See below |
| `gguf_quant`        | string, e.g. `"Q6_K"`                          | —           | Only when `quantization:"gguf"` |
| `sparse_structure` / `shape` / `texture` | `{steps,guidance_strength,guidance_rescale,rescale_t}` | pipeline defaults | Per-stage sampler overrides |
| `preview_fps`       | int                                            | `15`        | Only used with the `preview` format |

**Format notes:** `glb` carries the full PBR material (base color, metallic,
roughness, opacity). `obj` is zipped with its `.mtl` + texture. `ply` keeps
geometry with per-vertex colors. `stl` is geometry only. `preview` is a turntable
`.mp4` (slower — it re-renders the asset; needs an HDRI, see `TRELLIS2_PREVIEW_HDRI`).

## GGUF quantization

Set `quantization` to source the diffusion-transformer (DiT) weights from
[`Aero-Ex/Trellis2-GGUF`](https://huggingface.co/Aero-Ex/Trellis2-GGUF) instead of
the native bf16 weights:

- `"gguf-q8"` → `Q8_0` (near-lossless)
- `"gguf-q4"` → `Q4_K_M` (smaller, some quality loss)
- `"gguf"` + `gguf_quant:"Q6_K"` → any quant level the repo publishes

You can also pin the server's default with `TRELLIS2_DEFAULT_VARIANT` (`bf16`,
`gguf:Q8_0`, `gguf:Q4_K_M`).

> ⚠️ **What GGUF buys you here.** This service **dequantizes GGUF on load**, so the
> win is a smaller **download/disk** footprint (and you can run straight from the
> GGUF repo). Runtime **VRAM is roughly the same as bf16**, because weights expand
> back to bf16 once on the GPU. True VRAM savings require the ComfyUI-GGUF runtime
> (a different serving stack), which keeps weights quantized and dequantizes per
> matmul. On a DGX Spark (128 GB unified memory) you are not VRAM-constrained, so
> bf16 is the recommended default there. See `api/gguf_loader.py` for details.

> Switching `quantization` between requests forces an expensive model reload (only
> one variant is resident at a time). Prefer pinning a default and keeping it. Set
> `TRELLIS2_ALLOW_VARIANT_SWITCH=false` to reject per-request switches outright.

## Environment variables

| Var                            | Default                  | Purpose |
|--------------------------------|--------------------------|---------|
| `TRELLIS2_BASE_REPO`           | `microsoft/TRELLIS.2-4B` | Native weights repo |
| `TRELLIS2_GGUF_REPO`           | `Aero-Ex/Trellis2-GGUF`  | GGUF weights repo |
| `TRELLIS2_DEFAULT_VARIANT`     | `bf16`                   | Variant loaded at startup |
| `TRELLIS2_ALLOW_VARIANT_SWITCH`| `true`                   | Allow per-request variant changes |
| `TRELLIS2_PRELOAD`             | `true`                   | Load model on startup vs first request |
| `TRELLIS2_PREVIEW_HDRI`        | `assets/hdri/forest.exr` | HDRI for preview renders |
| `API_PORT` / `API_HOST`        | `8000` / `0.0.0.0`       | Bind address |
| `HF_TOKEN`                     | —                        | Required for gated model downloads |
