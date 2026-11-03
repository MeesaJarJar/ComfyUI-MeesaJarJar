"""JarJarSplatToMesh - gaussian splat PLY -> textured mesh file (OBJ/GLB) with decimation.

Pipeline (all local, no Hunyuan / no cloud):
  1) Load: reuse load_splat_ply_as_mesh (Poisson via pymeshlab, vertex_colors kept).
  2) Decimate (optional): pymeshlab quadric edge collapse down to target_faces,
     then re-transfer colors from the pre-decimation verts via scipy cKDTree
     (same pattern as the Poisson simplify step in splat_ply_nodes).
  3) Texture bake (optional): xatlas UV unwrap + software rasterization of the
     vertex colors into a texture image -> TextureVisuals, so the OBJ ships
     with a real .mtl + texture PNG (not just vertex colors).
  4) Export: trimesh .obj / .glb / .gltf to output_folder.

Requires: plyfile, scipy, trimesh (load path). pymeshlab (meshing + decimation),
xatlas + Pillow (texture bake). Bake/decimate steps degrade gracefully to a
vertex-color-only export when their optional dep is missing.
"""

import hashlib
import logging
import os

import numpy as np
import trimesh
from comfy_api.latest import IO
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

from .splat_ply_nodes import (
    ensure_outward_winding,
    load_splat_bytes_as_mesh,
    load_splat_ply_as_mesh,
)

log = logging.getLogger("jarjar")

_EXPORT_FORMATS = ("obj", "glb", "gltf")


def _cleanup_faces(mesh):
    """Drop degenerate / duplicate faces in place. Never raises."""
    try:
        mask = mesh.nondegenerate_faces()
        if not mask.all():
            mesh.update_faces(mask)
    except Exception:
        pass
    try:
        mask = mesh.unique_faces()
        if not mask.all():
            mesh.update_faces(mask)
    except Exception:
        pass
    return mesh


def _transfer_colors(tree_xyz, tree_rgba, verts):
    """Nearest-neighbor RGBA from a (xyz, rgba) cloud onto verts. Solid alpha."""
    tree = cKDTree(np.asarray(tree_xyz, dtype=np.float64))
    _, nn_idx = tree.query(np.asarray(verts, dtype=np.float64), k=1)
    out = np.asarray(tree_rgba)[nn_idx].copy()
    out[:, 3] = 255
    return out


def decimate_mesh(mesh, target_faces=50000, preserve_topology=True, quality_threshold=0.3):
    """Quadric decimate to ~target_faces. Colors re-transferred. Returns mesh.

    No-op (same object) when the mesh already has <= target faces.
    Raises RuntimeError/ImportError on failure so the caller can fall back.
    """
    import pymeshlab

    target_faces = max(1000, int(target_faces))
    try:
        n_faces = len(mesh.faces)
    except Exception:
        n_faces = 0
    if n_faces <= target_faces:
        log.info("SplatToMesh decimate: %d faces <= target %d, skipping",
                 n_faces, target_faces)
        return mesh

    src_xyz = np.asarray(mesh.vertices, dtype=np.float64)
    src_rgba = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=src_xyz,
                               face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preservetopology=bool(preserve_topology),
        qualitythr=float(quality_threshold),
    )
    sm = ms.current_mesh()
    verts = np.asarray(sm.vertex_matrix(), dtype=np.float32)
    faces = np.asarray(sm.face_matrix(), dtype=np.int64)
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise RuntimeError("decimation produced an empty mesh")

    rgba = _transfer_colors(src_xyz, src_rgba, verts)
    out = trimesh.Trimesh(vertices=verts, faces=faces,
                          vertex_colors=rgba, process=True)
    _cleanup_faces(out)
    log.info("SplatToMesh decimated: %d -> %d faces (target %d)",
             n_faces, len(out.faces), target_faces)
    return out


def _rasterize_texture(px, cols, size):
    """Barycentric fill of per-vertex colors into a (size x size) texture.

    px: (V,2) pixel coords, x=col left->right, y=row top->bottom.
    cols: (V,3) float32 0..1. Returns (uint8 HWC, painted bool mask).
    """
    h = w = int(size)
    tex = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    for t in range(0, px.shape[0] - 2, 3):
        tri = px[t:t + 3]
        c = cols[t:t + 3]
        lo = np.floor(tri.min(axis=0)).astype(int)
        hi = np.ceil(tri.max(axis=0)).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array([w - 1, h - 1], dtype=int))
        if hi[0] < lo[0] or hi[1] < lo[1]:
            continue
        xs = np.arange(lo[0], hi[0] + 1)
        ys = np.arange(lo[1], hi[1] + 1)
        xx, yy = np.meshgrid(xs, ys)
        den = ((tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0])
               + (tri[2, 0] - tri[1, 0]) * (tri[0, 1] - tri[2, 1]))
        if abs(float(den)) < 1e-12:
            continue
        l0 = ((tri[1, 1] - tri[2, 1]) * (xx - tri[2, 0])
              + (tri[2, 0] - tri[1, 0]) * (yy - tri[2, 1])) / den
        l1 = ((tri[2, 1] - tri[0, 1]) * (xx - tri[2, 0])
              + (tri[0, 0] - tri[2, 0]) * (yy - tri[2, 1])) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        rgb = (l0[..., None] * c[0] + l1[..., None] * c[1] + l2[..., None] * c[2])
        sub = tex[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        sub[inside] = rgb[inside]
        msub = mask[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        msub[inside] = True
    return tex, mask


def bake_vertex_texture(mesh, texture_size=2048):
    """UV-unwrap via xatlas and bake vertex colors to a texture image.

    Returns (mesh_with_TextureVisuals, PIL image). Raises on failure.
    Convention: uv v=0 at mesh bottom == image bottom (OBJ/Blender style);
    pixel row = (1 - v) * (H - 1).
    """
    import xatlas

    size = int(texture_size)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vc = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
    cols = (vc[:, :3].astype(np.float32) / 255.0)

    vmapping, indices, uvs = xatlas.parametrize(verts, faces)
    bverts = verts[vmapping].astype(np.float32)
    btris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    bcols = cols[vmapping]
    uvs = np.clip(np.asarray(uvs, dtype=np.float64), 0.0, 1.0)

    px = np.stack([uvs[:, 0] * (size - 1),
                   (1.0 - uvs[:, 1]) * (size - 1)], axis=1)
    flat_px = px[btris.reshape(-1)].reshape(-1, 2)
    flat_cols = bcols[btris.reshape(-1)].reshape(-1, 3)
    tex, mask = _rasterize_texture(flat_px, flat_cols, size)

    if not mask.any():
        raise RuntimeError("texture bake painted zero pixels")
    if not mask.all():
        # Nearest-painted-pixel fill for unpainted texels (seam bleeding).
        ys, xs = np.nonzero(mask)
        uy, ux = np.nonzero(~mask)
        if len(ux):
            filled = NearestNDInterpolator(
                np.stack([xs, ys], axis=1).astype(np.float64),
                tex[ys, xs].astype(np.float64),
            )(ux.astype(np.float64), uy.astype(np.float64))
            tex[uy, ux] = filled.astype(np.float32)

    img = Image.fromarray(np.round(np.clip(tex, 0.0, 1.0) * 255).astype(np.uint8), "RGB")
    buv = uvs  # one uv per baked vert (vmapping order == bverts order)
    visual = TextureVisuals(uv=buv.astype(np.float32), image=img,
                            material=SimpleMaterial(image=img))
    baked = trimesh.Trimesh(vertices=bverts, faces=btris,
                            visual=visual, process=True)
    _cleanup_faces(baked)
    log.info("SplatToMesh baked %dx%d texture (%.1f%% atlas coverage)",
             size, size, 100.0 * mask.mean())
    return baked, img


class JarJarSplatToMesh(IO.ComfyNode):
    """Splat PLY -> decimated, textured mesh file (OBJ/GLB). No Hunyuan needed."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSplatToMesh",
            display_name="JarJar: Splat PLY to Textured Mesh",
            category="JarJar/3D",
            description=(
                "Load a 3D gaussian splat PLY (from model_3d or file path), "
                "reconstruct a Poisson mesh, optionally decimate it (quadric "
                "edge collapse) and bake the splat colors to a real texture "
                "image (xatlas + rasterize), then save OBJ (with .mtl + "
                "texture PNG) / GLB / glTF. FBX is not offered: no local "
                "FBX writer is installed (use Tripo Convert or Blender for "
                "FBX). Without bake, color is carried as vertex colors only."
            ),
            inputs=[
                # NOTE: file_path stays REQUIRED (with an empty default, so it
                # never forces a connection) and FIRST on purpose. Optional
                # widgets render after required ones, so flipping file_path to
                # optional once scrambled every saved workflow's widget values.
                # Link-type inputs (model_3d) take no widget slot and are safe
                # anywhere; file_path keeps its original slot.
                IO.String.Input(
                    "file_path",
                    default="",
                    multiline=False,
                    tooltip="Absolute path to the splat PLY, e.g. C:\\path\\to\\model.ply. "
                            "Ignored when model_3d is connected.",
                ),
                IO.MultiType.Input(
                    IO.File3DAny.Input("model_3d"),
                    types=[IO.File3DSplatAny, IO.File3DPLY, IO.File3DSPLAT,
                           IO.File3DKSPLAT, IO.File3DSPZ],
                    optional=True,
                    tooltip="model_3d straight from Create 3D File (from Splat) "
                            "or a 3D load node. Wins over file_path when connected. "
                            "Must carry ply data (set the upstream node to ply); "
                            "ksplat/spz/splat are rejected.",
                ),
                IO.Combo.Input(
                    "format",
                    options=list(_EXPORT_FORMATS),
                    tooltip="obj: mesh + .mtl + texture PNG. glb/gltf: single self-contained file.",
                ),
                IO.String.Input(
                    "output_folder",
                    default="./output",
                    multiline=False,
                    tooltip="Folder the mesh file is written to (created if missing).",
                ),
                IO.String.Input(
                    "filename",
                    default="",
                    multiline=False,
                    placeholder="optional_name.obj",
                    tooltip="Optional file name (extension forced to format). Empty = <ply_stem>_mesh.<ext>.",
                ),
                IO.Float.Input(
                    "opacity_cutoff",
                    default=0.05, min=0.0, max=1.0, step=0.01,
                    tooltip="Drop gaussians with sigmoid(opacity) below this.",
                ),
                IO.Int.Input(
                    "poisson_depth",
                    default=9, min=6, max=12, step=1,
                    tooltip="Poisson reconstruction depth. Higher = more detail, slower.",
                ),
                IO.Int.Input(
                    "max_points",
                    default=100000, min=2000, max=500000, step=1000,
                    tooltip="Subsample limit for Poisson input (deterministic seed 0).",
                ),
                IO.Float.Input(
                    "rotate_x", default=0.0, min=-180.0, max=180.0, step=1.0,
                    tooltip="Rotate degrees about X, applied before meshing.",
                ),
                IO.Float.Input(
                    "rotate_y", default=0.0, min=-180.0, max=180.0, step=1.0,
                    tooltip="Rotate degrees about Y, applied before meshing.",
                ),
                IO.Float.Input(
                    "rotate_z", default=0.0, min=-180.0, max=180.0, step=1.0,
                    tooltip="Rotate degrees about Z, applied before meshing.",
                ),
                IO.Boolean.Input(
                    "decimate_enable",
                    default=True,
                    tooltip="Quadric decimate down to target_faces (skipped if already smaller).",
                ),
                IO.Int.Input(
                    "target_faces",
                    default=50000, min=1000, max=1000000, step=1000,
                    tooltip="Decimation target face count. Never upscales.",
                ),
                IO.Boolean.Input(
                    "preserve_topology",
                    default=True,
                    tooltip="Decimation: keep mesh topology (safer for rigging).",
                ),
                IO.Float.Input(
                    "quality_threshold",
                    default=0.3, min=0.0, max=1.0, step=0.05,
                    tooltip="Decimation quality threshold (pymeshlab qualitythr).",
                ),
                IO.Boolean.Input(
                    "bake_texture",
                    default=True,
                    tooltip="Bake vertex colors to a texture image (xatlas unwrap). Off = vertex colors only.",
                ),
                IO.Int.Input(
                    "texture_size",
                    default=2048, min=256, max=4096, step=256,
                    tooltip="Baked square texture resolution in px.",
                ),
                # NOTE: appended LAST on purpose. Required widgets map to saved
                # workflows positionally, so new widgets must only ever be
                # added at the end (see file_path comment above).
                IO.Boolean.Input(
                    "flip_faces",
                    default=False,
                    tooltip="Manually reverse face winding (inside-out <-> outside). "
                            "The node already auto-fixes inside-out meshes; use this "
                            "only if a viewer still shows the interior.",
                ),
            ],
            outputs=[
                IO.Custom("TRIMESH").Output(display_name="mesh"),
                IO.String.Output(display_name="filepath"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        file_path = (kwargs.get("file_path") or "").strip()
        sig = tuple(kwargs.get(k) for k in (
            "format", "opacity_cutoff", "poisson_depth", "max_points",
            "rotate_x", "rotate_y", "rotate_z", "decimate_enable",
            "target_faces", "preserve_topology", "quality_threshold",
            "flip_faces", "bake_texture", "texture_size", "filename",
            "output_folder"))
        model_3d = kwargs.get("model_3d")
        if model_3d is not None:
            try:
                data = bytes(model_3d.get_bytes())
                return (len(data), hashlib.sha256(data).hexdigest()) + sig
            except Exception:
                return ("model_3d",) + sig
        try:
            if file_path and os.path.exists(file_path):
                return (os.path.getmtime(file_path),) + sig
        except Exception:
            pass
        return (file_path,) + sig

    @classmethod
    def execute(cls, file_path="", model_3d=None, format="obj",
                output_folder="./output",
                filename="", opacity_cutoff=0.05, poisson_depth=9,
                max_points=100000, rotate_x=0.0, rotate_y=0.0, rotate_z=0.0,
                decimate_enable=True, target_faces=50000,
                preserve_topology=True, quality_threshold=0.3,
                flip_faces=False,
                bake_texture=True, texture_size=2048) -> IO.NodeOutput:
        import comfy.model_management as mm

        splat_data = None
        if model_3d is not None:
            fmt = (getattr(model_3d, "format", "") or "").lower()
            if fmt not in ("ply",):
                raise ValueError(
                    "JarJarSplatToMesh: model_3d carries {!r}, need ply. "
                    "Set Create 3D File (from Splat) to ply format.".format(fmt))
            try:
                splat_data = bytes(model_3d.get_bytes())
            except Exception as e:
                raise ValueError(
                    "JarJarSplatToMesh: could not read model_3d bytes: {}".format(e))
            if not splat_data:
                raise ValueError("JarJarSplatToMesh: model_3d is empty.")
            if (file_path or "").strip():
                log.info("SplatToMesh: model_3d connected, ignoring file_path")

        file_path = (file_path or "").strip()
        if (file_path.startswith('"') and file_path.endswith('"')) or \
           (file_path.startswith("'") and file_path.endswith("'")):
            file_path = file_path[1:-1].strip()

        mesh_kwargs = dict(
            opacity_cutoff=float(opacity_cutoff),
            poisson_depth=int(poisson_depth),
            max_points=int(max_points),
            rotate_x=float(rotate_x),
            rotate_y=float(rotate_y),
            rotate_z=float(rotate_z),
        )
        if splat_data is not None:
            mesh, error = load_splat_bytes_as_mesh(splat_data, **mesh_kwargs)
        else:
            if not file_path:
                raise ValueError(
                    "JarJarSplatToMesh: connect model_3d or provide file_path.")
            if not os.path.exists(file_path):
                raise ValueError("File not found: {}".format(file_path))
            if os.path.splitext(file_path)[1].lower() != ".ply":
                raise ValueError(
                    "JarJarSplatToMesh needs a .ply splat file, got: {}".format(file_path))
            mesh, error = load_splat_ply_as_mesh(file_path, **mesh_kwargs)

        if mesh is None:
            raise ValueError("Failed to load splat PLY: {}".format(error))

        mm.throw_exception_if_processing_interrupted()

        n_faces = len(mesh.faces) if getattr(mesh, "faces", None) is not None else 0
        log.info("SplatToMesh loaded: %s verts, %s faces", len(mesh.vertices), n_faces)

        if n_faces > 0 and decimate_enable:
            try:
                mesh = decimate_mesh(mesh, target_faces=int(target_faces),
                                     preserve_topology=bool(preserve_topology),
                                     quality_threshold=float(quality_threshold))
            except ImportError:
                log.warning("SplatToMesh: pymeshlab missing, decimation skipped")
            except Exception as e:
                log.warning("SplatToMesh: decimation failed (%s), keeping full mesh", e)
            mm.throw_exception_if_processing_interrupted()

        has_vc = False
        try:
            vc = mesh.visual.vertex_colors
            has_vc = vc is not None and len(vc) == len(mesh.vertices)
        except Exception:
            pass

        n_faces = len(mesh.faces) if getattr(mesh, "faces", None) is not None else 0
        if bake_texture and n_faces > 0 and has_vc:
            try:
                mesh, _img = bake_vertex_texture(mesh, texture_size=int(texture_size))
            except ImportError as e:
                log.warning("SplatToMesh: bake skipped (missing %s), vertex colors only", e)
            except Exception as e:
                log.warning("SplatToMesh: texture bake failed (%s), vertex colors only", e)
            mm.throw_exception_if_processing_interrupted()
        elif bake_texture:
            log.info("SplatToMesh: bake skipped (faces=%d colors=%s)", n_faces, has_vc)

        # Final orientation safety net (decimate/bake preserve winding, but
        # never ship an inside-out mesh): auto-fix, then manual override.
        ensure_outward_winding(mesh, label="SplatToMesh")
        if flip_faces and getattr(mesh, "faces", None) is not None \
                and len(mesh.faces) > 0:
            mesh.faces = np.ascontiguousarray(
                np.asarray(mesh.faces)[:, ::-1])
            log.info("SplatToMesh: manual flip_faces applied")

        out_dir = (output_folder or "./output").strip() or "./output"
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(file_path))[0] or "splat_model"
        name = (filename or "").strip() or "{}_mesh.{}".format(stem, format)
        if os.path.splitext(name)[1].lower() != "." + format:
            name = os.path.splitext(name)[0] + "." + format
        out_path = os.path.join(out_dir, name)

        mesh.export(out_path, file_type=format)
        log.info("SplatToMesh saved %s (%d verts, %d faces) -> %s",
                 format, len(mesh.vertices), len(mesh.faces), out_path)
        return IO.NodeOutput(mesh, out_path)


NODE_CLASS_MAPPINGS = {
    "JarJarSplatToMesh": JarJarSplatToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarSplatToMesh": "JarJar: Splat PLY to Textured Mesh",
}
