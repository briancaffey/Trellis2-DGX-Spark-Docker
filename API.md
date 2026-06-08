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

| Method | Path                                | Body / use                                                  |
|--------|-------------------------------------|-------------------------------------------------------------|
| GET    | `/health`                           | —                                                           |
| POST   | `/generate`                         | `multipart/form-data`: `image` file + optional `options` JSON |
| POST   | `/generate/base64`                  | `application/json`: `{ "image_base64": "...", "options": {…} }` |
| GET    | `/results/{job_id}`                 | Manifest (file list + URLs) for a `files`-mode result       |
| GET    | `/results/{job_id}/files/{name}`    | Download one **raw, unzipped** file                         |

Full interactive OpenAPI docs are at `/docs` (Swagger) and `/redoc`; the raw spec
is at `/openapi.json`.

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

## Recommended workflow (save GLB + view in three.js)

For a backend that forwards a request to this node, stores the result, and renders
it client-side, **request GLB and take the raw bytes** — it's a single
self-contained file with full PBR materials and loads directly in three.js
`GLTFLoader` (no `.mtl`/texture juggling like OBJ).

```bash
# one call in, raw GLB out — stream the body straight into S3/your file store
curl -X POST http://<node>:8000/generate \
  -F image=@input.png \
  -F 'options={"formats":["glb"]}' \
  -o asset.glb
# generation metadata (seed, vertices, faces, timing, variant) is in the
# X-Trellis-Metadata response header — handy to persist alongside the file.
```

GLB is the **superset** format: store it as your canonical artifact. You can derive
a (lossy) OBJ/PLY/STL from it later, but converting *up* from OBJ can't recover the
metallic/roughness/opacity that only GLB preserves. If you do need another format,
request it at generation time (`"formats":["glb","obj"]`) so each is exported from
the same full-fidelity mesh.

## `options` reference

| Field               | Type / values                                  | Default     | Notes |
|---------------------|------------------------------------------------|-------------|-------|
| `seed`              | int                                            | `42`        | |
| `randomize_seed`    | bool                                           | `false`     | Ignores `seed` when true |
| `resolution`        | `"512"` \| `"1024"` \| `"1536"`                | `"1024"`    | Higher = sharper, slower, more VRAM |
| `formats`           | list of `glb,obj,ply,stl,preview`              | `["glb"]`   | See packaging below |
| `response_mode`     | `"auto"` \| `"zip"` \| `"files"` \| `"json"`   | `"auto"`    | How files are packaged — see below |
| `decimation_target` | int                                            | `500000`    | Target face count for the exported mesh |
| `texture_size`      | int                                            | `2048`      | Baked texture resolution |
| `quantization`      | `"bf16"` \| `"gguf-q8"` \| `"gguf-q4"` \| `"gguf"` | server default | See below |
| `gguf_quant`        | string, e.g. `"Q6_K"`                          | —           | Only when `quantization:"gguf"` |
| `sparse_structure` / `shape` / `texture` | `{steps,guidance_strength,guidance_rescale,rescale_t}` | pipeline defaults | Per-stage sampler overrides |
| `preview_fps`       | int                                            | `15`        | Only used with the `preview` format |

**Format notes:** `glb` carries the full PBR material (base color, metallic,
roughness, opacity). `obj` is three files (`.obj` + `.mtl` + texture `.png`). `ply`
keeps geometry with per-vertex colors. `stl` is geometry only. `preview` is a
turntable `.mp4` (slower — it re-renders the asset; needs an HDRI, see
`TRELLIS2_PREVIEW_HDRI`).

### Packaging (`response_mode`)

| Value     | What you get |
|-----------|--------------|
| `auto`    | (default) exactly one output file ⇒ that file raw; multiple files ⇒ a single `.zip` |
| `zip`     | Always a single `.zip` (all files loose inside, plus `metadata.json`) |
| `files`   | **Raw, unzipped.** Returns a JSON manifest with a download URL per file; fetch each with `GET /results/{job_id}/files/{name}`. Files are held in memory (default 1 h / 2 GB cap). Great for testing individual files. |
| `json`    | One JSON body with every file base64-encoded inline + metadata |

`binary` is accepted as a deprecated alias for `auto`. Generation metadata is also
returned in the `X-Trellis-Metadata` response header on raw/zip responses.

#### `files` mode example (download the raw OBJ pieces individually)

```bash
# 1) generate, ask for unzipped files
curl -s -X POST http://localhost:8000/generate \
  -F image=@chair.png \
  -F 'options={"formats":["obj"],"response_mode":"files"}' | tee manifest.json

# 2) the manifest lists model.obj / material.mtl / material_0.png, each with a url:
#    { "job_id":"...", "files":[{"filename":"model.obj","url":".../files/model.obj"}, ...] }

# 3) pull any single raw file
curl -OJ http://localhost:8000/results/<job_id>/files/model.obj

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
| `TRELLIS2_RESULTS_TTL_SEC`     | `3600`                   | How long `files`-mode results stay downloadable |
| `TRELLIS2_RESULTS_MAX_BYTES`   | `2147483648` (2 GB)      | In-memory cap for `files`-mode results |
| `API_PORT` / `API_HOST`        | `8000` / `0.0.0.0`       | Bind address |
| `HF_TOKEN`                     | —                        | Required for gated model downloads |
