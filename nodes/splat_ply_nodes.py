"""JarJarLoadSplatPly - Load a 3D Gaussian Splat PLY (no faces) as a textured trimesh.

Splat PLYs (e.g. from TripoSplat / SplatToFile3D / SuperSplat exports) contain only
vertices with properties x,y,z (+optional nx,ny,nz), f_dc_0/1/2 (SH base color),
opacity (pre-sigmoid logit), scale_*, rot_*. They have 0 faces, so trimesh/UniRig/
Hunyuan Paint all reject them.

This loader mirrors the SPZ path in classic_load_mesh.py:
  1) Parse PLY -> xyz + RGB (0.5 + C0*f_dc) + alpha (sigmoid(opacity)).
  2) Filter by opacity cutoff + finite positions.
  3) Poisson path (pymeshlab, screened poisson depth=9): estimate normals, reconstruct
     watertight mesh, transfer nearest-neighbor vertex_colors (opaque) via scipy cKDTree.
     Gives real faces for rigging / UV unwrap / Hunyuan Paint.
  4) Fallback: point-cloud Trimesh with 0 faces but full RGBA vertex_colors
     (previewable / GLB-exportable as points, same as the SPZ fallback).

Requires: plyfile, scipy, trimesh. Poisson path additionally requires pymeshlab.
"""

import logging
import os
from io import BytesIO

import numpy as np
import trimesh
from comfy_api.latest import io
from plyfile import PlyData
from scipy.spatial import cKDTree

log = logging.getLogger("jarjar")

_SH_C0 = 0.28209479177387814  # SH band-0 constant: DC -> base RGB


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _signed_volume(verts, faces):
    """Signed mesh volume (>0 = outward winding). 0 faces -> 0.0."""
    if faces is None or len(faces) == 0:
        return 0.0
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces)
    return float(np.sum(np.einsum("ij,ij->i", v[f[:, 0]],
                                  np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0)


def ensure_outward_winding(mesh, label="mesh"):
    """Flip face winding if a watertight mesh is inside-out (volume < 0).

    pymeshlab's point-cloud normals can orient inward, which makes Poisson
    emit inward-wound faces (viewers with backface culling then show the
    interior, lit from the wrong side). Returns True if flipped.
    """
    try:
        if getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
            return False
        if not bool(mesh.is_watertight):
            return False
        if _signed_volume(mesh.vertices, mesh.faces) < 0:
            mesh.faces = np.ascontiguousarray(
                np.asarray(mesh.faces)[:, ::-1])
            log.info("%s: inside-out winding detected, faces flipped", label)
            return True
    except Exception as e:
        log.warning("%s: winding check failed (%s)", label, e)
    return False


def _euler_matrix(rx_deg, ry_deg, rz_deg):
    """Euler rotation matrix (degrees) applied as R = Rz @ Ry @ Rx."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _splat_arrays_from_vertex(v):
    """Vertex record array -> (xyz (N,3) float32, rgba (N,4) uint8)."""
    names = set(v.dtype.names or [])
    for req in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"):
        if req not in names:
            raise ValueError(
                "Not a gaussian-splat PLY: missing '{}' "
                "(need x,y,z,f_dc_0/1/2,opacity)".format(req)
            )

    xyz = np.stack([np.asarray(v["x"], dtype=np.float64),
                    np.asarray(v["y"], dtype=np.float64),
                    np.asarray(v["z"], dtype=np.float64)], axis=1).astype(np.float32)
    fdc = np.stack([np.asarray(v["f_dc_0"], dtype=np.float32),
                    np.asarray(v["f_dc_1"], dtype=np.float32),
                    np.asarray(v["f_dc_2"], dtype=np.float32)], axis=1)
    rgb = np.clip(0.5 + _SH_C0 * fdc, 0.0, 1.0)
    alpha = _sigmoid(np.asarray(v["opacity"], dtype=np.float32))

    rgba = np.concatenate([np.round(rgb * 255).astype(np.uint8),
                           np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint8)[:, None]], axis=1)
    return xyz, rgba


def _read_splat_ply(file_path: str):
    """Read gaussian-splat PLY from disk. Returns (xyz, rgba)."""
    ply = PlyData.read(file_path)
    if "vertex" not in [e.name for e in ply.elements]:
        raise ValueError("PLY has no 'vertex' element")
    return _splat_arrays_from_vertex(ply["vertex"].data)


def _read_splat_ply_bytes(data: bytes):
    """Read gaussian-splat PLY from in-memory bytes. Returns (xyz, rgba)."""
    ply = PlyData.read(BytesIO(bytes(data)))
    if "vertex" not in [e.name for e in ply.elements]:
        raise ValueError("PLY has no 'vertex' element")
    return _splat_arrays_from_vertex(ply["vertex"].data)


def _poisson_via_pymeshlab(xyz, rgba, depth=9, max_points=100000, seed=0):
    """Reconstruct watertight mesh with pymeshlab. Returns trimesh or raises."""
    import pymeshlab

    n = xyz.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max_points, replace=False)
        xyz = xyz[keep]
        rgba = rgba[keep]
        n = max_points
        log.info("Splat PLY poisson: subsampled to %d points", n)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(xyz, dtype=np.float64)))

    ms.compute_normal_for_point_clouds(k=20)
    ms.generate_surface_reconstruction_screened_poisson(depth=int(depth))

    pm = ms.current_mesh()
    verts = np.asarray(pm.vertex_matrix(), dtype=np.float32)
    faces = np.asarray(pm.face_matrix(), dtype=np.int64)
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise RuntimeError("Poisson produced an empty mesh")

    tree = cKDTree(np.asarray(xyz, dtype=np.float64))
    _, nn_idx = tree.query(np.asarray(verts, dtype=np.float64), k=1)
    transferred = rgba[nn_idx].copy()
    transferred[:, 3] = 255  # solid mesh surface

    if len(faces) > 200000:
        log.info("Splat PLY poisson simplifying %d -> ~80k tris", len(faces))
        ms2 = pymeshlab.MeshSet()
        ms2.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(verts, dtype=np.float64),
                                    face_matrix=np.asarray(faces, dtype=np.int32)))
        ms2.meshing_decimation_quadric_edge_collapse(targetfacenum=80000)
        sm = ms2.current_mesh()
        verts = np.asarray(sm.vertex_matrix(), dtype=np.float32)
        faces = np.asarray(sm.face_matrix(), dtype=np.int64)
        _, nn_idx = tree.query(np.asarray(verts, dtype=np.float64), k=1)
        transferred = rgba[nn_idx].copy()
        transferred[:, 3] = 255

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_colors=transferred, process=True)
    try:
        mask = mesh.nondegenerate_faces()
        if not mask.all():
            mesh.update_faces(mask)
        mask = mesh.unique_faces()
        if not mask.all():
            mesh.update_faces(mask)
    except Exception:
        pass
    return mesh


def _mesh_from_splat_arrays(xyz, rgba, label, file_path="", file_name="",
                              opacity_cutoff=0.05, poisson_depth=9,
                              max_points=100000,
                              rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
    """Shared filter -> Poisson -> fallback core. Returns (mesh, error)."""
    n_orig = xyz.shape[0]
    alpha = rgba[:, 3].astype(np.float32) / 255.0
    valid = np.isfinite(xyz).all(axis=1) & (alpha > float(opacity_cutoff))
    xyz = xyz[valid]
    rgba = rgba[valid]
    n = xyz.shape[0]
    log.info("Splat PLY %s: %d / %d gaussians kept (opacity>%.3f)",
             label, n, n_orig, opacity_cutoff)
    if n == 0:
        return None, "Splat PLY has no visible gaussians after opacity filter"
    if rotate_x or rotate_y or rotate_z:
        R = _euler_matrix(float(rotate_x), float(rotate_y), float(rotate_z))
        xyz = (np.asarray(xyz, dtype=np.float64) @ R.T).astype(np.float32)
        log.info("Splat PLY rotated: rx=%.1f ry=%.1f rz=%.1f deg",
                 rotate_x, rotate_y, rotate_z)
    if n < 2000:
        log.info("Splat PLY: only %d points, skipping Poisson (need 2000+)", n)
    else:
        try:
            import pymeshlab  # noqa: F401
            mesh = _poisson_via_pymeshlab(xyz, rgba, depth=poisson_depth,
                                          max_points=max_points)
            ensure_outward_winding(mesh, label="Splat PLY {}".format(label))
            mesh.metadata["file_path"] = file_path
            mesh.metadata["file_name"] = file_name
            mesh.metadata["file_format"] = ".ply"
            mesh.metadata["splat_gaussians"] = n_orig
            mesh.metadata["splat_vertices"] = n
            log.info("Splat PLY poisson final: %d verts %d faces with vertex_colors",
                     len(mesh.vertices), len(mesh.faces))
            return mesh, ""
        except ImportError:
            log.info("pymeshlab not available, splat PLY loads as point cloud")
        except Exception as e:
            log.warning("Splat PLY poisson failed (%s), point-cloud fallback", e)

    mesh = trimesh.Trimesh(vertices=np.asarray(xyz, dtype=np.float64),
                           faces=np.zeros((0, 3), dtype=np.int64),
                           vertex_colors=rgba, process=False)
    mesh.metadata["file_path"] = file_path
    mesh.metadata["file_name"] = file_name
    mesh.metadata["file_format"] = ".ply"
    mesh.metadata["splat_gaussians"] = n_orig
    mesh.metadata["splat_vertices"] = n
    mesh.metadata["is_point_cloud"] = True
    log.info("Splat PLY point-cloud fallback: %d verts 0 faces with vertex_colors",
             len(mesh.vertices))
    return mesh, ""


def load_splat_ply_as_mesh(file_path: str, opacity_cutoff=0.05,
                           poisson_depth=9, max_points=100000,
                           rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
    """Load splat PLY -> (mesh, error_message). Never raises."""
    try:
        xyz, rgba = _read_splat_ply(file_path)
    except Exception as e:
        return None, "Error reading splat PLY: {}".format(e)
    return _mesh_from_splat_arrays(
        xyz, rgba, os.path.basename(file_path),
        file_path=file_path, file_name=os.path.basename(file_path),
        opacity_cutoff=opacity_cutoff, poisson_depth=poisson_depth,
        max_points=max_points,
        rotate_x=rotate_x, rotate_y=rotate_y, rotate_z=rotate_z)


def load_splat_bytes_as_mesh(data: bytes, opacity_cutoff=0.05,
                             poisson_depth=9, max_points=100000,
                             rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
    """Load splat PLY from in-memory bytes -> (mesh, error_message). Never raises."""
    try:
        xyz, rgba = _read_splat_ply_bytes(data)
    except Exception as e:
        return None, "Error reading splat PLY bytes: {}".format(e)
    return _mesh_from_splat_arrays(
        xyz, rgba, "model_3d",
        file_path="", file_name="model_3d",
        opacity_cutoff=opacity_cutoff, poisson_depth=poisson_depth,
        max_points=max_points,
        rotate_x=rotate_x, rotate_y=rotate_y, rotate_z=rotate_z)


def _mm():
    import comfy.model_management
    return comfy.model_management


class JarJarLoadSplatPly(io.ComfyNode):
    """Load a 3D Gaussian Splat PLY (0 faces) as a textured mesh.

    Give the full path to a splat PLY (x,y,z,f_dc_*,opacity,scale_*,rot_*).
    Reconstructs a watertight Poisson mesh (pymeshlab) that keeps the splat
    colors as vertex_colors, with faces for rigging / UV unwrap / Hunyuan Paint.
    rotate_x/y/z (degrees, R = Rz @ Ry @ Rx) fix orientation up front, applied
    to the points before meshing so faces, normals and colors all agree.
    Without enough points (or if Poisson fails) it returns a colored point cloud
    Trimesh instead. Outputs TRIMESH compatible with UniRig nodes.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JarJarLoadSplatPly",
            display_name="JarJar: Load Splat PLY as Mesh",
            category="JarJar/3D",
            description=(
                "Load a 3D Gaussian Splat PLY (points only, no faces) and convert it "
                "to a textured mesh. Base color comes from f_dc (SH DC), alpha from "
                "sigmoid(opacity). rotate_x/y/z fix orientation before meshing. "
                "Poisson reconstruction (pymeshlab) produces faces "
                "for rigging/UV/Hunyuan Paint and preserves color as vertex_colors; "
                "on failure it falls back to a colored point cloud."
            ),
            inputs=[
                io.String.Input(
                    "file_path",
                    default="",
                    multiline=False,
                    tooltip="Absolute path to the splat PLY, e.g. C:\\path\\to\\model.ply",
                ),
                io.Float.Input(
                    "opacity_cutoff",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Drop gaussians with sigmoid(opacity) below this (0-1).",
                ),
                io.Int.Input(
                    "poisson_depth",
                    default=9,
                    min=6,
                    max=12,
                    step=1,
                    tooltip="Poisson reconstruction depth. Higher = more detail, slower.",
                ),
                io.Int.Input(
                    "max_points",
                    default=100000,
                    min=2000,
                    max=500000,
                    step=1000,
                    tooltip="Subsample limit for Poisson input (deterministic seed 0).",
                ),
                io.Float.Input(
                    "rotate_x",
                    default=0.0,
                    min=-180.0,
                    max=180.0,
                    step=1.0,
                    tooltip="Rotate degrees about X, applied to points before meshing. Order: Rz @ Ry @ Rx.",
                ),
                io.Float.Input(
                    "rotate_y",
                    default=0.0,
                    min=-180.0,
                    max=180.0,
                    step=1.0,
                    tooltip="Rotate degrees about Y, applied to points before meshing. Order: Rz @ Ry @ Rx.",
                ),
                io.Float.Input(
                    "rotate_z",
                    default=0.0,
                    min=-180.0,
                    max=180.0,
                    step=1.0,
                    tooltip="Rotate degrees about Z, applied to points before meshing. Order: Rz @ Ry @ Rx.",
                ),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        file_path = (kwargs.get("file_path") or "").strip()
        if file_path and os.path.exists(file_path):
            return (os.path.getmtime(file_path), kwargs.get("opacity_cutoff"),
                    kwargs.get("poisson_depth"), kwargs.get("max_points"),
                    kwargs.get("rotate_x"), kwargs.get("rotate_y"), kwargs.get("rotate_z"))
        return file_path

    @classmethod
    def execute(cls, file_path, opacity_cutoff=0.05, poisson_depth=9, max_points=100000,
                rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
        file_path = (file_path or "").strip()
        if not file_path:
            raise ValueError("File path cannot be empty. Provide the full absolute path to a splat PLY file.")
        if (file_path.startswith('"') and file_path.endswith('"')) or \
           (file_path.startswith("'") and file_path.endswith("'")):
            file_path = file_path[1:-1].strip()
        if not os.path.exists(file_path):
            raise ValueError("File not found: {}".format(file_path))
        if os.path.splitext(file_path)[1].lower() != ".ply":
            raise ValueError("JarJarLoadSplatPly needs a .ply splat file, got: {}".format(file_path))

        _mm().throw_exception_if_processing_interrupted()

        mesh, error = load_splat_ply_as_mesh(
            file_path,
            opacity_cutoff=float(opacity_cutoff),
            poisson_depth=int(poisson_depth),
            max_points=int(max_points),
            rotate_x=float(rotate_x),
            rotate_y=float(rotate_y),
            rotate_z=float(rotate_z),
        )
        if mesh is None:
            raise ValueError("Failed to load splat PLY: {}".format(error))

        try:
            n_faces = len(getattr(mesh, "faces", []))
        except Exception:
            n_faces = 0
        log.info("Loaded splat PLY: %s vertices, %s faces", len(mesh.vertices), n_faces)

        return io.NodeOutput(mesh)


NODE_CLASS_MAPPINGS = {
    "JarJarLoadSplatPly": JarJarLoadSplatPly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarLoadSplatPly": "JarJar: Load Splat PLY as Mesh",
}
