"""Convert a generated ``trimesh.Trimesh`` into downloadable files.

Each format yields one or more raw files as ``(filename, media_type, bytes)``.
The GLB is the source of truth: it carries the full PBR material (base color,
metallic, roughness, opacity) baked by ``o_voxel.postprocess.to_glb``. The other
formats are lossy w.r.t. materials and are provided for convenience:

  * obj  - geometry + UVs + base-color texture (three loose files: .obj/.mtl/.png).
  * ply  - geometry with the texture sampled down to per-vertex colors.
  * stl  - geometry only (no color at all).
"""
import os
import tempfile
from typing import List, Tuple

# (filename, media_type, bytes)
FileTuple = Tuple[str, str, bytes]


def _glb_bytes(mesh) -> bytes:
    try:
        # Newer trimesh encodes textures as WebP, matching the reference app.py.
        return mesh.export(file_type="glb", extension_webp=True)
    except TypeError:
        return mesh.export(file_type="glb")


def _ply_bytes(mesh) -> bytes:
    m = mesh.copy()
    try:
        # Collapse UV texture to per-vertex colors so PLY retains some color.
        m.visual = m.visual.to_color()
    except Exception:
        pass
    out = m.export(file_type="ply")
    return out.encode() if isinstance(out, str) else out


def _stl_bytes(mesh) -> bytes:
    out = mesh.export(file_type="stl")
    return out.encode() if isinstance(out, str) else out


def _obj_files(mesh) -> List[FileTuple]:
    """Export OBJ to a temp dir (so the .mtl + texture get written) and read the
    loose files back. Returns the .obj, its .mtl, and any texture images."""
    media_by_ext = {
        ".obj": "text/plain; charset=utf-8",
        ".mtl": "text/plain; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    out: List[FileTuple] = []
    with tempfile.TemporaryDirectory() as d:
        mesh.export(os.path.join(d, "model.obj"))
        for fn in sorted(os.listdir(d)):
            ext = os.path.splitext(fn)[1].lower()
            with open(os.path.join(d, fn), "rb") as fh:
                out.append((fn, media_by_ext.get(ext, "application/octet-stream"), fh.read()))
    return out


def export_files(mesh, fmt: str) -> List[FileTuple]:
    """Return the raw file(s) for a single requested format."""
    if fmt == "glb":
        return [("model.glb", "model/gltf-binary", _glb_bytes(mesh))]
    if fmt == "ply":
        return [("model.ply", "application/octet-stream", _ply_bytes(mesh))]
    if fmt == "stl":
        return [("model.stl", "model/stl", _stl_bytes(mesh))]
    if fmt == "obj":
        return _obj_files(mesh)
    raise ValueError(f"Unsupported mesh format: {fmt}")
