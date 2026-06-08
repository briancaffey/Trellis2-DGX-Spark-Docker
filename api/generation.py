"""Runs the pipeline and turns its output mesh into the requested files.

This mirrors the reference extract path in ``TRELLIS.2/app.py`` (sampler params ->
``pipeline.run`` -> ``o_voxel.postprocess.to_glb``) rather than inventing a new one.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from collections import OrderedDict

import numpy as np
import torch

from . import config, exporters
from .schemas import GenerateOptions

# AABB used by the reference app for the unit-cube assets.
_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
_MAX_SEED = np.iinfo(np.int32).max
_envmap = None


def _resolve_seed(options: GenerateOptions) -> int:
    if options.randomize_seed:
        return int(np.random.randint(0, _MAX_SEED))
    return options.seed


def _get_envmap():
    """Lazily build the HDRI environment map used for preview renders."""
    global _envmap
    if _envmap is None:
        import cv2
        from trellis2.renderers import EnvMap

        img = cv2.imread(config.PREVIEW_HDRI, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(
                f"Preview HDRI not found at '{config.PREVIEW_HDRI}'. Set TRELLIS2_PREVIEW_HDRI "
                f"or drop 'preview' from the requested formats."
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _envmap = EnvMap(torch.tensor(img, dtype=torch.float32, device="cuda"))
    return _envmap


def _render_preview(mesh, fps: int) -> bytes:
    import os as _os
    import tempfile
    import imageio
    from trellis2.utils import render_utils

    mesh.simplify(16777216)  # nvdiffrast vertex limit (matches example.py)
    frames = render_utils.make_pbr_vis_frames(
        render_utils.render_video(mesh, envmap=_get_envmap())
    )
    # imageio's ffmpeg writer wants a real file path, not a file-like object.
    with tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "preview.mp4")
        imageio.mimsave(path, frames, fps=fps)
        with open(path, "rb") as fh:
            return fh.read()


def generate(pipeline, image, options: GenerateOptions):
    """Generate a 3D asset. Returns (metadata: dict, files: OrderedDict[name -> (media_type, bytes)])."""
    import o_voxel

    seed = _resolve_seed(options)
    texture_size = min(options.texture_size, config.MAX_TEXTURE_SIZE)
    decimation_target = min(options.decimation_target, config.MAX_DECIMATION_TARGET)

    t0 = time.time()
    outputs, latents = pipeline.run(
        image,
        seed=seed,
        sparse_structure_sampler_params=options.sparse_structure.to_params(),
        shape_slat_sampler_params=options.shape.to_params(),
        tex_slat_sampler_params=options.texture.to_params(),
        pipeline_type=options.pipeline_type,
        return_latent=True,
    )
    mesh = outputs[0]
    res = latents[2]
    t_sample = time.time() - t0

    files = OrderedDict()
    mesh_formats = [f for f in options.formats if f != "preview"]
    textured = None
    t_bake = 0.0

    # Bake the textured mesh (PBR) only if a mesh format is requested. This is the
    # GLB source-of-truth. Done before any preview render (which mutates `mesh`).
    if mesh_formats:
        t1 = time.time()
        textured = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=_AABB,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
        )
        t_bake = time.time() - t1
        for fmt in mesh_formats:
            data = exporters.export_mesh(textured, fmt)
            files[f"model.{exporters.FILE_EXT[fmt]}"] = (exporters.MEDIA_TYPES[fmt], data)

    if "preview" in options.formats:
        data = _render_preview(mesh, options.preview_fps)
        files["preview.mp4"] = (exporters.MEDIA_TYPES["preview"], data)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = {
        "seed": seed,
        "resolution": options.resolution,
        "pipeline_type": options.pipeline_type,
        "variant": getattr(pipeline, "_active_variant", None),
        "texture_size": texture_size,
        "decimation_target": decimation_target,
        "formats": options.formats,
        "timing_sec": {"sample": round(t_sample, 2), "bake": round(t_bake, 2)},
    }
    if textured is not None:
        metadata["vertices"] = int(textured.vertices.shape[0])
        metadata["faces"] = int(textured.faces.shape[0])
    return metadata, files
