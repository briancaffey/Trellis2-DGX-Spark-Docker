"""Pydantic request/response models for the API."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


Resolution = Literal["512", "1024", "1536"]
OutputFormat = Literal["glb", "obj", "ply", "stl", "preview"]

# How the generated file(s) come back:
#   auto   - one file -> raw download; multiple -> a single .zip   (default)
#   zip    - always a single .zip (loose files inside)
#   files  - raw, UNZIPPED: returns a JSON manifest with a download URL per file
#            (files are held in memory briefly; good for testing individual files)
#   json   - a single JSON body with each file base64-encoded inline
#   binary - deprecated alias for "auto"
ResponseMode = Literal["auto", "zip", "files", "json", "binary"]

# Accepted quantization values. "bf16" = native weights. The "gguf:*" values map
# to files in the Aero-Ex/Trellis2-GGUF repo. We expose two friendly aliases plus
# a passthrough for any GGUF quant string the repo happens to publish.
Quantization = Literal["bf16", "gguf-q8", "gguf-q4", "gguf"]


class SamplerOverrides(BaseModel):
    """Optional per-stage sampler overrides. Anything left as None falls back to
    the pipeline's built-in defaults (see pipeline.json on the model repo)."""
    steps: Optional[int] = Field(None, ge=1, le=50)
    guidance_strength: Optional[float] = Field(None, ge=1.0, le=10.0)
    guidance_rescale: Optional[float] = Field(None, ge=0.0, le=1.0)
    rescale_t: Optional[float] = Field(None, ge=1.0, le=6.0)

    def to_params(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class GenerateOptions(BaseModel):
    """Everything that controls a single generation, except the image itself."""

    seed: int = 42
    randomize_seed: bool = False
    resolution: Resolution = "1024"

    # Postprocessing / mesh extraction.
    decimation_target: int = Field(500_000, ge=10_000)
    texture_size: int = Field(2048, ge=512)

    # Output.
    formats: List[OutputFormat] = Field(
        default_factory=lambda: ["glb"],
        description="One or more of: glb, obj, ply, stl, preview (turntable mp4).",
    )
    response_mode: ResponseMode = Field(
        "auto",
        description="auto = raw if single file else zip; zip = always zip; "
                    "files = unzipped per-file download URLs (JSON manifest); "
                    "json = base64 inline.",
    )

    # Model selection. None -> use the server's configured default variant.
    quantization: Optional[Quantization] = None
    # When quantization is the generic "gguf", this picks the exact quant level,
    # e.g. "Q8_0", "Q6_K", "Q4_K_M". Ignored for the gguf-q8 / gguf-q4 aliases.
    gguf_quant: Optional[str] = None

    # Advanced sampler controls (optional).
    sparse_structure: SamplerOverrides = Field(default_factory=SamplerOverrides)
    shape: SamplerOverrides = Field(default_factory=SamplerOverrides)
    texture: SamplerOverrides = Field(default_factory=SamplerOverrides)

    # Turntable preview video (only used when "preview" is in formats).
    preview_fps: int = Field(15, ge=1, le=60)

    @property
    def pipeline_type(self) -> str:
        # Mirror app.py: 512 is single-pass, the higher resolutions use cascades.
        return {
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[self.resolution]


class Base64GenerateRequest(BaseModel):
    """Body for the JSON variant of the generate endpoint."""
    image_base64: str = Field(..., description="Base64-encoded image (optionally a data: URI).")
    options: GenerateOptions = Field(default_factory=GenerateOptions)


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    device: Optional[str] = None
    loaded_variant: Optional[str] = None
    default_variant: str
    allow_variant_switch: bool


class FilePayload(BaseModel):
    filename: str
    media_type: str
    base64: str


class JsonGenerateResponse(BaseModel):
    """response_mode="json": every file inlined as base64."""
    metadata: dict
    files: List[FilePayload]


class FileInfo(BaseModel):
    """One downloadable file in a `files`-mode manifest."""
    filename: str
    media_type: str
    size_bytes: int
    url: str = Field(..., description="GET this for the raw, unzipped file bytes.")


class ResultManifest(BaseModel):
    """response_mode="files": metadata + per-file download URLs (no zip)."""
    job_id: str
    metadata: dict
    files: List[FileInfo]
    expires_in_sec: int

