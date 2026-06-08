"""FastAPI service for TRELLIS.2 image-to-3D generation.

This package wraps the native ``Trellis2ImageTo3DPipeline`` (see
``TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py``) behind a small HTTP API
that accepts an image and returns generated 3D assets (GLB / OBJ / PLY / STL and
an optional turntable preview video).

It also supports loading the diffusion-transformer (DiT) weights from the
community GGUF quantization at https://huggingface.co/Aero-Ex/Trellis2-GGUF via a
``quantization`` flag on the request. See ``api/gguf_loader.py`` for the details
and the important caveats around what GGUF does and does not buy you here.
"""
