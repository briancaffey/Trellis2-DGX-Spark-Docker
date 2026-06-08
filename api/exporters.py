"""Convert a generated ``trimesh.Trimesh`` into the various downloadable formats.

The GLB is the source of truth: it carries the full PBR material (base color,
metallic, roughness, opacity) baked by ``o_voxel.postprocess.to_glb``. The other
formats are lossy w.r.t. materials and are provided for convenience:

  * obj  - geometry + UVs + base-color texture (zipped: .obj/.mtl/texture).
  * ply  - geometry with the texture sampled down to per-vertex colors.
  * stl  - geometry only (no color at all).
"""
import io
import os
import tempfile
import zipfile


MEDIA_TYPES = {
    "glb": "model/gltf-binary",
    "obj": "application/zip",
    "ply": "application/octet-stream",
    "stl": "model/stl",
    "preview": "video/mp4",
}

# What each format is actually written as on disk / in the response.
FILE_EXT = {"glb": "glb", "obj": "zip", "ply": "ply", "stl": "stl", "preview": "mp4"}


def glb_bytes(mesh) -> bytes:
    try:
        # Newer trimesh encodes textures as WebP, matching the reference app.py.
        return mesh.export(file_type="glb", extension_webp=True)
    except TypeError:
        return mesh.export(file_type="glb")


def obj_zip_bytes(mesh) -> bytes:
    """Export OBJ to a temp dir (so the .mtl + texture get written) and zip it."""
    with tempfile.TemporaryDirectory() as d:
        mesh.export(os.path.join(d, "model.obj"))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn in sorted(os.listdir(d)):
                zf.write(os.path.join(d, fn), arcname=fn)
        return buf.getvalue()


def ply_bytes(mesh) -> bytes:
    m = mesh.copy()
    try:
        # Collapse UV texture to per-vertex colors so PLY retains some color.
        m.visual = m.visual.to_color()
    except Exception:
        pass
    out = m.export(file_type="ply")
    return out.encode() if isinstance(out, str) else out


def stl_bytes(mesh) -> bytes:
    out = mesh.export(file_type="stl")
    return out.encode() if isinstance(out, str) else out


_EXPORTERS = {
    "glb": glb_bytes,
    "obj": obj_zip_bytes,
    "ply": ply_bytes,
    "stl": stl_bytes,
}


def export_mesh(mesh, fmt: str) -> bytes:
    if fmt not in _EXPORTERS:
        raise ValueError(f"Unsupported mesh format: {fmt}")
    return _EXPORTERS[fmt](mesh)
