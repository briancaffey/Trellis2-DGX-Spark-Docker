# TRELLIS.2 Docker (DGX SPARK)

Docker image optimized for DGX SPARK.

Original repository: [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)

### Quick Start
```bash
# Initialize submodules
git submodule update --init --recursive

# Create .env with your Hugging Face token and user IDs
echo "HF_TOKEN=your_huggingface_token_here" > .env
echo "UID=$(id -u)" >> .env
echo "GID=$(id -g)" >> .env

# Launch
docker compose up --build
```

**Mandatory:** Request access to these models on Hugging Face:
- [DINOv3](https://huggingface.co/collections/facebook/dinov3)
- [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

Access: `http://localhost:7860`

### Alternative Apps
```bash
# Run texturing app instead of default
APP_SCRIPT=app_texturing.py docker compose up

# Or set in .env
echo "APP_SCRIPT=app_texturing.py" >> .env
```

### HTTP API (FastAPI)
A REST service for programmatic image-to-3D generation (returns GLB/OBJ/PLY/STL,
optionally GGUF-quantized weights). See [API.md](API.md).
```bash
docker compose up --build api          # http://localhost:8000  (docs at /docs)
curl -X POST http://localhost:8000/generate -F image=@chair.png -o chair.glb
```

**Note:** First build takes time (CUDA compilation + model downloads).
