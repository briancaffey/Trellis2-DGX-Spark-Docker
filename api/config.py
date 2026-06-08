"""Server configuration, driven entirely by environment variables."""
import os


# Hugging Face repo for the native (bf16/fp16) TRELLIS.2 weights.
BASE_MODEL_REPO = os.getenv("TRELLIS2_BASE_REPO", "microsoft/TRELLIS.2-4B")

# Hugging Face repo for the community GGUF quantization.
GGUF_MODEL_REPO = os.getenv("TRELLIS2_GGUF_REPO", "Aero-Ex/Trellis2-GGUF")

# Which variant to load at startup: "bf16" (native) or a "gguf:<QUANT>" string,
# e.g. "gguf:Q8_0" or "gguf:Q4_K_M". Requests may override this per-call, but
# switching variants forces an expensive pipeline reload (see pipeline_manager).
DEFAULT_VARIANT = os.getenv("TRELLIS2_DEFAULT_VARIANT", "bf16")

# Allow requests to switch the model variant at runtime. When false, the server
# loads DEFAULT_VARIANT once and rejects requests that ask for a different one.
ALLOW_VARIANT_SWITCH = os.getenv("TRELLIS2_ALLOW_VARIANT_SWITCH", "true").lower() in ("1", "true", "yes")

# Eagerly load the default variant on startup instead of on first request.
PRELOAD = os.getenv("TRELLIS2_PRELOAD", "true").lower() in ("1", "true", "yes")

# Cap the number of formats / safety limits.
MAX_TEXTURE_SIZE = int(os.getenv("TRELLIS2_MAX_TEXTURE_SIZE", "4096"))
MAX_DECIMATION_TARGET = int(os.getenv("TRELLIS2_MAX_DECIMATION_TARGET", "1000000"))

# Directory for transient output files (cleaned per request).
WORK_DIR = os.getenv("TRELLIS2_WORK_DIR", "/tmp/trellis2_api")

# In-memory results store (used by response_mode="files" for per-file downloads).
RESULTS_TTL_SEC = int(os.getenv("TRELLIS2_RESULTS_TTL_SEC", "3600"))
RESULTS_MAX_BYTES = int(os.getenv("TRELLIS2_RESULTS_MAX_BYTES", str(2 * 1024 ** 3)))

# HDRI used for the optional turntable preview render. Relative to the TRELLIS.2
# working directory (the default app ships these under assets/hdri/).
PREVIEW_HDRI = os.getenv("TRELLIS2_PREVIEW_HDRI", "assets/hdri/forest.exr")
