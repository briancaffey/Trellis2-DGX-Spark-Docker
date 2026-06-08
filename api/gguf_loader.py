"""Load TRELLIS.2 diffusion-transformer (DiT) weights from the community GGUF
quantization (https://huggingface.co/Aero-Ex/Trellis2-GGUF).

----------------------------------------------------------------------------
IMPORTANT — what GGUF does and does not buy you here
----------------------------------------------------------------------------
The GGUF files are primarily intended for the ComfyUI-GGUF runtime, which keeps
weights quantized in memory and dequantizes them on the fly per matmul. That is
what produces a smaller *VRAM* footprint.

This loader takes the simpler, robust path: it **dequantizes on load** and hands
the native PyTorch pipeline ordinary fp16/bf16 tensors. Consequences:

  * ✅ Smaller *download / disk* footprint (e.g. a Q4_K_M DiT is a fraction of the
    bf16 safetensors), and you can run straight from the GGUF repo.
  * ⚠️  Runtime *VRAM* is essentially the same as bf16, because the weights are
    expanded back to bf16 once resident on the GPU.

If you need true VRAM savings you want the ComfyUI-GGUF node graph (a different
serving stack), not this native pipeline. ``_dequantize_on_load`` is the seam
where a future on-the-fly-dequant (GGMLTensor-style) implementation would slot in.

----------------------------------------------------------------------------
Robustness notes (this code path can only be exercised on a CUDA box with the
weights present, so it is written defensively):
  * GGUF weight *filenames* are resolved by listing the repo and fuzzy-matching
    the checkpoint basename + quant level, rather than hard-coding paths.
  * GGUF tensor *names* are matched against the model's own state_dict keys, and
    each tensor is reshaped to the expected shape — so we don't depend on GGUF's
    (reversed) dim ordering being identical to PyTorch's.
"""
from typing import Dict, Optional
import json
import re

import numpy as np
import torch

import trellis2.models as tmodels
from trellis2.pipelines import Trellis2ImageTo3DPipeline


# Model keys whose weights we try to source from GGUF. Everything else (the small
# VAE decoders/encoders, the DINOv3 image conditioner, the rembg model) is loaded
# natively at fp16 — those are not the memory-dominant parts and are not always
# published as GGUF.
FLOW_MODEL_TAGS = (
    "ss_flow",            # sparse_structure_flow_model
    "img2shape",          # shape_slat_flow_model_{512,1024}
    "imgshape2tex",       # tex_slat_flow_model_{512,1024}
)

# Friendly aliases -> concrete GGUF quant level.
QUANT_ALIASES = {
    "gguf-q8": "Q8_0",
    "gguf-q4": "Q4_K_M",
}


def resolve_quant(quantization: str, gguf_quant: Optional[str]) -> str:
    """Map a request's quantization choice to a concrete GGUF quant string."""
    if quantization in QUANT_ALIASES:
        return QUANT_ALIASES[quantization]
    if quantization == "gguf":
        if not gguf_quant:
            raise ValueError("quantization='gguf' requires 'gguf_quant' (e.g. 'Q8_0', 'Q4_K_M').")
        return gguf_quant
    raise ValueError(f"Not a GGUF quantization: {quantization!r}")


def _checkpoint_basename(model_path: str) -> str:
    """'microsoft/TRELLIS.2-4B/ckpts/slat_flow_..._1024_bf16' -> 'slat_flow_..._1024'."""
    name = model_path.split("/")[-1]
    # Drop a trailing precision tag if present (bf16/fp16/fp8/...).
    return re.sub(r"_(bf16|fp16|fp8|f16|f32)$", "", name)


def _find_gguf_file(gguf_repo: str, basename: str, quant: str) -> str:
    """Find the GGUF file in ``gguf_repo`` for this checkpoint + quant level.

    We list the repo and pick the best match rather than guessing a path, since
    the repo's exact folder/quant naming is the one thing we can't verify offline.
    """
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(gguf_repo) if f.endswith(".gguf")]
    base_key = basename.lower()
    quant_key = quant.lower()

    # Prefer files that mention both the checkpoint basename and the quant level.
    scored = []
    for f in files:
        fl = f.lower()
        score = 0
        if base_key in fl:
            score += 2
        # token-overlap fallback for slightly different basenames
        score += sum(1 for tok in base_key.split("_") if len(tok) > 2 and tok in fl) * 0.1
        if quant_key in fl:
            score += 2
        scored.append((score, f))
    scored.sort(reverse=True)

    if not scored or scored[0][0] < 4:
        avail = "\n  ".join(files) or "(none)"
        raise FileNotFoundError(
            f"Could not confidently match a GGUF file for checkpoint '{basename}' "
            f"at quant '{quant}' in repo '{gguf_repo}'.\nAvailable .gguf files:\n  {avail}"
        )
    return scored[0][1]


def _normalize_key(k: str) -> str:
    """Normalize a tensor name for matching (strip common prefixes/suffixes)."""
    k = k.strip()
    for pref in ("model.", "module.", "net.", "dit.", "transformer."):
        if k.startswith(pref):
            k = k[len(pref):]
    return k


def _dequantize_on_load(reader_tensor) -> np.ndarray:
    """Return a tensor's values as a float32 numpy array, dequantizing if needed.

    This is the seam for a future on-the-fly (memory-saving) implementation: today
    we materialize full-precision values here.
    """
    try:
        from gguf.quants import dequantize as gguf_dequantize
        return gguf_dequantize(reader_tensor.data, reader_tensor.tensor_type).astype(np.float32)
    except Exception:
        # F16/F32/BF16 (and anything already unpacked) arrive as a plain array.
        return np.asarray(reader_tensor.data, dtype=np.float32)


def _load_gguf_state_dict(gguf_path: str, expected: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Read + dequantize a GGUF file into a state_dict matching ``expected``.

    Each GGUF tensor is matched to an expected key and reshaped to the expected
    shape, sidestepping GGUF's reversed dim-order convention.
    """
    from gguf import GGUFReader

    # Index expected keys by themselves and by a normalized form for fuzzy matching.
    norm_index: Dict[str, str] = {}
    for key in expected:
        norm_index.setdefault(_normalize_key(key), key)

    reader = GGUFReader(gguf_path)
    out: Dict[str, torch.Tensor] = {}
    unmatched = []
    for t in reader.tensors:
        name = t.name
        target = name if name in expected else norm_index.get(_normalize_key(name))
        if target is None:
            unmatched.append(name)
            continue
        exp = expected[target]
        values = _dequantize_on_load(t)
        if values.size != exp.numel():
            unmatched.append(f"{name} (numel {values.size} != {exp.numel()})")
            continue
        tensor = torch.from_numpy(values).reshape(exp.shape).to(exp.dtype)
        out[target] = tensor

    if unmatched:
        print(f"[gguf] {len(unmatched)} GGUF tensor(s) not mapped: "
              + ", ".join(unmatched[:8]) + (" ..." if len(unmatched) > 8 else ""))
    missing = [k for k in expected if k not in out]
    if missing:
        print(f"[gguf] {len(missing)} model param(s) not found in GGUF (kept as initialized): "
              + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    return out


def _load_model_from_gguf(model_path: str, gguf_repo: str, quant: str) -> torch.nn.Module:
    """Build a flow model from its native config and load dequantized GGUF weights."""
    from huggingface_hub import hf_hub_download

    # 1. Build the module from its native JSON config (config is tiny; we never
    #    download the big bf16 safetensors for these flow models).
    path_parts = model_path.split("/")
    repo_id = "/".join(path_parts[:2])
    model_name = "/".join(path_parts[2:])
    config_file = hf_hub_download(repo_id, f"{model_name}.json")
    with open(config_file) as f:
        config = json.load(f)
    model = getattr(tmodels, config["name"])(**config["args"])

    # 2. Resolve + download the matching GGUF weights, then load them in.
    basename = _checkpoint_basename(model_path)
    gguf_file = _find_gguf_file(gguf_repo, basename, quant)
    gguf_path = hf_hub_download(gguf_repo, gguf_file)
    print(f"[gguf] {basename}: loading {gguf_file} ({quant})")

    state_dict = _load_gguf_state_dict(gguf_path, model.state_dict())
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def load_gguf_pipeline(base_repo: str, gguf_repo: str, quant: str) -> Trellis2ImageTo3DPipeline:
    """Load the full pipeline, sourcing the DiT flow models from GGUF.

    Implementation note: we temporarily monkeypatch ``trellis2.models.from_pretrained``
    so the unmodified ``Trellis2ImageTo3DPipeline.from_pretrained`` does all of its
    normal work (samplers, normalization, image conditioner, rembg) while the flow
    models get sourced from GGUF. Everything else loads natively.
    """
    original = tmodels.from_pretrained

    def patched_from_pretrained(path: str, **kwargs):
        if any(tag in path for tag in FLOW_MODEL_TAGS):
            try:
                return _load_model_from_gguf(path, gguf_repo, quant)
            except Exception as exc:  # fall back to native weights for this model
                print(f"[gguf] falling back to native weights for {path}: {exc}")
        return original(path, **kwargs)

    tmodels.from_pretrained = patched_from_pretrained
    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(base_repo)
    finally:
        tmodels.from_pretrained = original
    return pipeline
