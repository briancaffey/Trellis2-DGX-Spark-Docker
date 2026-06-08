"""Owns the single in-process TRELLIS.2 pipeline.

Only one pipeline variant is kept resident at a time (a 4B model barely fits as
it is). Switching variants tears down the old one, frees the GPU, and loads the
new one. All generation is serialized through a lock — the model uses the whole
GPU and ``pipeline.run`` is not reentrant.
"""
import gc
import threading
from typing import Optional

import torch

from . import config
from .gguf_loader import load_gguf_pipeline, resolve_quant


def _normalize_variant(quantization: Optional[str], gguf_quant: Optional[str]) -> str:
    """Turn a request's (quantization, gguf_quant) into a canonical variant key.

    Accepts request-style values ("bf16", "gguf-q8", "gguf-q4", "gguf"+gguf_quant)
    as well as already-canonical keys ("bf16", "gguf:Q8_0") as used by the
    TRELLIS2_DEFAULT_VARIANT env var.
    """
    if quantization is None or quantization == "bf16":
        return "bf16"
    if quantization.startswith("gguf:"):  # already canonical, e.g. "gguf:Q4_K_M"
        return quantization
    quant = resolve_quant(quantization, gguf_quant)  # validates + raises on bad input
    return f"gguf:{quant}"


class PipelineManager:
    def __init__(self):
        self._pipeline = None
        self._variant: Optional[str] = None
        self._load_lock = threading.Lock()   # guards (re)loading the model
        self._run_lock = threading.Lock()    # serializes generation

    @property
    def loaded_variant(self) -> Optional[str]:
        return self._variant

    @property
    def device(self) -> Optional[str]:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self, variant: str):
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        print(f"[pipeline] loading variant '{variant}' ...")
        if variant == "bf16":
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(config.BASE_MODEL_REPO)
        else:
            quant = variant.split(":", 1)[1]
            pipeline = load_gguf_pipeline(config.BASE_MODEL_REPO, config.GGUF_MODEL_REPO, quant)
        pipeline.cuda()
        self._pipeline = pipeline
        self._variant = variant
        print(f"[pipeline] variant '{variant}' ready.")

    def _unload(self):
        self._pipeline = None
        self._variant = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure(self, variant: str):
        """Make sure ``variant`` is the resident pipeline, (re)loading if needed."""
        if self._variant == variant and self._pipeline is not None:
            return
        with self._load_lock:
            if self._variant == variant and self._pipeline is not None:
                return
            if not config.ALLOW_VARIANT_SWITCH and self._variant is not None:
                raise RuntimeError(
                    f"Variant switching is disabled (server is pinned to "
                    f"'{self._variant}'). Requested '{variant}'."
                )
            if self._pipeline is not None:
                self._unload()
            self._load(variant)

    def get(self, quantization: Optional[str], gguf_quant: Optional[str]):
        """Resolve the request's variant, ensure it's loaded, and return the pipeline."""
        variant = _normalize_variant(quantization or config.DEFAULT_VARIANT, gguf_quant)
        self.ensure(variant)
        return self._pipeline

    def preload(self):
        self.ensure(_normalize_variant(config.DEFAULT_VARIANT, None))

    @property
    def run_lock(self) -> threading.Lock:
        return self._run_lock


manager = PipelineManager()
