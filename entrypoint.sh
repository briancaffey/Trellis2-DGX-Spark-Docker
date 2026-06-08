#!/bin/bash
set -e

echo "========================================"
echo "TRELLIS.2 Docker Environment start"
echo "========================================"

# Ensure CUDA env
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$CUDA_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
export ATTN_BACKEND=${ATTN_BACKEND:-flash-attn}

echo "Repository ready at: $(pwd)"

# Print basic runtime info
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA Home: ${CUDA_HOME}"
echo "ATTN_BACKEND: ${ATTN_BACKEND}"

echo "========================================"
echo "Environment ready!"
echo "Working directory: $(pwd)"
echo "========================================"

# Determine which app to run
APP_SCRIPT=${APP_SCRIPT:-app.py}

# Run the specified app or custom command
if [ $# -gt 0 ]; then
  exec "$@"
elif [ "$APP_SCRIPT" = "api" ]; then
  # FastAPI service. The ./api package is mounted at /workspace/TRELLIS.2/api so
  # that both `api.server` and `trellis2` are importable from this working dir.
  API_HOST=${API_HOST:-${GRADIO_SERVER_NAME:-0.0.0.0}}
  API_PORT=${API_PORT:-${GRADIO_SERVER_PORT:-7860}}
  echo "Starting FastAPI service on ${API_HOST}:${API_PORT}"
  exec uvicorn api.server:app --host "$API_HOST" --port "$API_PORT" --workers 1
else
  exec python "$APP_SCRIPT"
fi