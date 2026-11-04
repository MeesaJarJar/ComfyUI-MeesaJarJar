"""JarJarLoadMesh - Load a mesh from an absolute file path (GLB/OBJ/PLY/STL/etc. + SPZ splat + splat PLY)."""

import os
import gzip
import struct
import logging
import trimesh
import numpy as np
from typing import Tuple, Optional
from comfy_api.latest import io

try:
    import igl
    _IGL_AVAILABLE = True
except Exception:
    igl = None
    _IGL_AVAILABLE = False

log = logging.getLogger("jarjar")

_C0 = 0.28209479177387814
_SPZ_MAGIC = 0x5053474E  # "NGSP"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _parse_spz_gaussian(data: bytes):
    raw = gzip.decompress(data)
    if struct.unpack_from('<I', raw, 0)[0] != _SPZ_MAGIC:
        raise ValueError("invalid .spz (bad magic)")
    version = struct.unpack_from('<I', raw, 4)[0]
    n = struct.unpack_from('<I', raw, 8)[0]
    frac_bits = raw[13]
    off = 16

    if version == 1:
        xyz = np.frombuffer(raw, '<f2', count=n * 3, offset=off).astype(np.float32).reshape(n, 3)
        off += n * 6
    elif version in (2, 3):
        b = np.frombuffer(raw, np.uint8, count=n * 9, offset=off).reshape(n, 3, 3).astype(np.int64)
        v = (b[..., 2] << 16) | (b[..., 1] << 8) | b[..., 0]
        v = np.where(v & 0x800000, v - 0x1000000, v)
        xyz = (v / (1 << frac_bits)).astype(np.float32)
        off += n * 9
    else:
        raise ValueError(f"unsupported .spz version {version}")

    alpha = np.frombuffer(raw, np.uint8, count=n, offset=off).astype(np.float32) / 255.0
    off += n
    cb = np.frombuffer(raw, np.uint8, count=n * 3, offset=off).reshape(n, 3).astype(np.float32)
    rgb = (cb / 255.0 - 0.5) * (_C0 / 0.15) + 0.5
    return xyz, np.clip(rgb, 0, 1), alpha


def _splat_points_to_mesh(xyz, rgb, alpha, file_path, file_format, label, extra_meta=None):
    xyz = np.asarray(xyz, dtype=np.float32).copy()
    xyz[:, 1] *= -1.0
    xyz[:, 2] *= -1.0
    n_orig = xyz.shape[0]
    valid = np.isfinite(xyz).all(axis=1) & (np.asarray(alpha).reshape(-1) > 0.025)
    xyz = xyz[valid]
    rgb = np.clip(rgb[valid], 0, 1)
    alpha = np.clip(np.asarray(alpha).reshape(-1)[valid], 0, 1)
    n = xyz.shape[0]
    if n == 0:
        return None, f"{label} has no visible splats after opacity filter"
    log.info("%s after filter: %d / %d splats kept (alpha>0.025)", label, n, n_orig)

    rgba = np.concatenate([np.round(rgb * 255).astype(np.uint8),
                           np.round(alpha * 255).astype(np.uint8)[:, None]], axis=1)

    try:
        import open3d as o3d
        if 2000 <= n <= 250000:
            xyz_mesh = xyz
            rgb_mesh = rgb
            if n > 100000:
                pcd_tmp = o3d.geometry.PointCloud()
                pcd_tmp.points = o3d.utility.Vector3dVector(xyz)
                pcd_tmp.colors = o3d.utility.Vector3dVector(rgb)
                bbox_diag = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
                vs = max(bbox_diag / 200.0, 1e-4)
                pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=vs)
                xyz_mesh = np.asarray(pcd_tmp.points, dtype=np.float32)
                rgb_mesh = np.asarray(pcd_tmp.colors, dtype=np.float32)
                log.info("%s poisson: downsampled %d -> %d (voxel_size=%.5f)", label, n, xyz_mesh.shape[0], vs)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_mesh.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb_mesh, 0, 1))

            bbox = xyz_mesh.max(axis=0) - xyz_mesh.min(axis=0)
            diag = float(np.linalg.norm(bbox))
            radius = max(diag * 0.015, 1e-4)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
            try:
                pcd.orient_normals_consistent_tangent_plane(100)
            except Exception:
                log.debug("%s orient normals skipped", label, exc_info=True)

            log.info("%s poisson: normals radius=%.5f diag=%.4f", label, radius, diag)
            mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.1, linear_fit=False)
            log.info("%s poisson raw: %d verts %d tris", label, len(mesh_o3d.vertices), len(mesh_o3d.triangles))

            verts = np.asarray(mesh_o3d.vertices, dtype=np.float32)
            if verts.shape[0] > 0 and len(mesh_o3d.triangles) > 0:
                try:
                    from scipy.spatial import cKDTree
                    _, nn_idx = cKDTree(xyz).query(verts, k=1)
                    transferred_rgba = rgba[nn_idx]
                    transferred_rgba[:, 3] = 255
                except Exception as e:
                    log.warning("%s color transfer failed (%s), using white", label, e)
                    transferred_rgba = np.full((verts.shape[0], 4), 255, dtype=np.uint8)

                faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)
                if len(faces) > 200000:
                    try:
                        log.info("%s poisson simplifying %d -> 80000 tris", label, len(faces))
                        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=80000)
                        verts = np.asarray(mesh_o3d.vertices, dtype=np.float32)
                        faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)
                        from scipy.spatial import cKDTree
                        _, nn_idx = cKDTree(xyz).query(verts, k=1)
                        transferred_rgba = rgba[nn_idx]
                        transferred_rgba[:, 3] = 255
                    except Exception as e:
                        log.warning("%s poisson decimation failed: %s", label, e)

                mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=transferred_rgba, process=True)
                try:
                    mask = mesh.nondegenerate_faces()
                    if not mask.all():
                        mesh.update_faces(mask)
                    mask = mesh.unique_faces()
                    if not mask.all():
                        mesh.update_faces(mask)
                except Exception:
                    log.debug("%s cleanup faces skipped", label, exc_info=True)
                mesh.metadata['file_path'] = file_path
                mesh.metadata['file_name'] = os.path.basename(file_path)
                mesh.metadata['file_format'] = file_format
                mesh.metadata['splat_points'] = n_orig
                mesh.metadata['splat_vertices'] = n
                if extra_meta:
                    mesh.metadata.update(extra_meta)
                log.info("%s poisson final: %d verts %d faces with vertex_colors", label, len(mesh.vertices), len(mesh.faces))
                return mesh, ""
            else:
                log.warning("%s poisson produced empty mesh, falling back to point cloud", label)
        else:
            log.info("%s poisson skipped (n=%d out of 2k..250k range), using point cloud", label, n)
    except ImportError:
        log.info("open3d not available, %s will be loaded as point cloud with vertex_colors", label)
    except Exception as e:
        log.warning("%s poisson failed (%s), falling back to point cloud", label, e, exc_info=True)

    mesh = trimesh.Trimesh(vertices=xyz, faces=np.zeros((0, 3), dtype=np.int64), vertex_colors=rgba, process=False)
    mesh.metadata['file_path'] = file_path
    mesh.metadata['file_name'] = os.path.basename(file_path)
    mesh.metadata['file_format'] = file_format
    mesh.metadata['splat_points'] = n_orig
    mesh.metadata['splat_vertices'] = n
    mesh.metadata['is_point_cloud'] = True
    if extra_meta:
        mesh.metadata.update(extra_meta)
    log.info("%s point-cloud fallback: %d verts 0 faces with vertex_colors", label, len(mesh.vertices))
    return mesh, ""


def _load_splat_ply_as_mesh(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    try:
        from plyfile import PlyData
    except ImportError:
        return None, "splat PLY needs the 'plyfile' package (pip install plyfile)"

    try:
        ply = PlyData.read(file_path)
        if 'vertex' not in ply:
            return None, "PLY has no vertex element"
        v = ply['vertex']
        names = set(v.data.dtype.names or [])
        for axis in ('x', 'y', 'z'):
            if axis not in names:
                return None, f"splat PLY missing '{axis}' property"
        xyz = np.vstack([np.asarray(v[axis], dtype=np.float32) for axis in ('x', 'y', 'z')]).T

        if all(k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2')):
            sh_dc = np.vstack([np.asarray(v[k], dtype=np.float32) for k in ('f_dc_0', 'f_dc_1', 'f_dc_2')]).T
            rgb = np.clip(sh_dc * _C0 + 0.5, 0, 1)
        elif all(k in names for k in ('red', 'green', 'blue')):
            rgb = np.vstack([np.asarray(v[k], dtype=np.float32) for k in ('red', 'green', 'blue')]).T
            if rgb.size == 0 or rgb.ndim != 2 or rgb.shape[1] != 3:
                rgb = np.full_like(xyz, 1.0)
            elif rgb.max(initial=0) <= 1.0:
                rgb = np.clip(rgb, 0, 1)
            else:
                rgb = np.clip(rgb / 255.0, 0, 1)
        else:
            log.warning("splat PLY has neither f_dc_* nor red/green/blue; using white")
            rgb = np.full_like(xyz, 1.0)

        if 'opacity' in names:
            alpha = _sigmoid(np.asarray(v['opacity'], dtype=np.float64)).astype(np.float32)
        else:
            alpha = np.ones((xyz.shape[0],), dtype=np.float32)

        log.info("splat PLY parsed: %d points from %s", xyz.shape[0], os.path.basename(file_path))
        return _splat_points_to_mesh(
            xyz.astype(np.float32), rgb.astype(np.float32), alpha,
            file_path, '.ply', 'splat PLY',
            extra_meta={'spz_splats': xyz.shape[0], 'spz_vertices': xyz.shape[0]},
        )
    except Exception as e:
        log.warning("splat PLY load failed: %s", e, exc_info=True)
        return None, f"Error loading splat PLY: {str(e)}"


def _load_spz_as_mesh(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    try:
        data = open(file_path, 'rb').read()
        xyz, rgb, alpha = _parse_spz_gaussian(data)
        n_orig = xyz.shape[0]
        log.info("SPZ parsed: %d splats from %s", n_orig, os.path.basename(file_path))
        mesh, err = _splat_points_to_mesh(
            np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.float32),
            np.asarray(alpha, dtype=np.float32),
            file_path, '.spz', 'SPZ',
        )
        if mesh is None:
            return None, err
        mesh.metadata['spz_splats'] = n_orig
        return mesh, ""
    except Exception as e:
        log.warning("SPZ load failed: %s", e, exc_info=True)
        return None, f"Error loading SPZ: {str(e)}"


def _mm():
    import comfy.model_management
    return comfy.model_management


def load_mesh_file(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    if not file_path or not file_path.strip():
        return None, "File path cannot be empty"

    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    log.info("File extension detected: '%s'", ext)

    if ext == ".spz":
        return _load_spz_as_mesh(file_path)

    try:
        log.info("Loading: %s", file_path)
        loaded = trimesh.load(file_path, process=False, maintain_order=True)
        log.info("Loaded type: %s", type(loaded).__name__)

        if isinstance(loaded, trimesh.Scene):
            log.info("Converting Scene to single mesh (scene has %s geometries)", len(loaded.geometry))
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if isinstance(mesh, trimesh.points.PointCloud):
            verts = np.asarray(mesh.vertices)
            cols = getattr(mesh, 'colors', None)
            if cols is not None:
                cols = np.asarray(cols)
                if cols.size == 0:
                    cols = None
            if cols is not None and len(cols) == len(verts):
                if cols.size and float(cols.max(initial=0)) <= 1.0:
                    cols = (cols * 255).astype(np.uint8)
                if cols.ndim == 2 and cols.shape[1] == 3:
                    cols = np.column_stack([cols, np.full(len(cols), 255, dtype=np.uint8)])
                mesh = trimesh.Trimesh(vertices=verts, faces=np.zeros((0, 3), dtype=np.int64), vertex_colors=cols, process=False)
            elif ext == ".ply":
                log.info("PointCloud has no colors, trying splat-PLY parser...")
                splat_mesh, splat_err = _load_splat_ply_as_mesh(file_path)
                if splat_mesh is not None:
                    return splat_mesh, ""
                log.warning("splat-PLY parser failed (%s), continuing with colorless cloud", splat_err)
                mesh = trimesh.Trimesh(vertices=verts, faces=np.zeros((0, 3), dtype=np.int64), process=False)
            else:
                mesh = trimesh.Trimesh(vertices=verts, faces=np.zeros((0, 3), dtype=np.int64), process=False)
            log.info("PointCloud converted to Trimesh: %d verts", len(mesh.vertices))

        if mesh is None or len(mesh.vertices) == 0:
            return None, f"Failed to read mesh or mesh is empty: {file_path}"

        is_point_cloud = len(getattr(mesh, 'faces', [])) == 0
        if is_point_cloud:
            has_colors = hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None
            log.info("Point-cloud mesh (0 faces) %s vertex_colors", "with" if has_colors else "without")
            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()
            log.info("Successfully loaded point-cloud: %s vertices, 0 faces", len(mesh.vertices))
            return mesh, ""

        log.info("Initial mesh: %s vertices, %s faces", len(mesh.vertices), len(mesh.faces))

        if hasattr(mesh, "faces") and len(mesh.faces) > 0 and mesh.faces.shape[1] != 3:
            log.warning("Mesh has non-triangular faces, re-wrapping as Trimesh...")
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False, maintain_order=True)

        valid_faces_mask = mesh.unique_faces() & mesh.nondegenerate_faces()
        if not valid_faces_mask.all():
            mesh.update_faces(valid_faces_mask)

        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

        log.info("Successfully loaded: %s vertices, %s faces", len(mesh.vertices), len(mesh.faces))
        return mesh, ""

    except Exception as e:
        if not _IGL_AVAILABLE:
            log.info("Trimesh failed: %s. (libigl fallback unavailable)", str(e))
            return None, f"Error loading mesh: {str(e)}"
        log.info("Trimesh failed: %s, trying libigl fallback...", str(e))
        try:
            v, f = igl.read_triangle_mesh(file_path)
            if v is None or f is None or len(v) == 0 or len(f) == 0:
                return None, f"Failed to read mesh: {file_path}"
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_order=True)
            valid_faces_mask = mesh.unique_faces() & mesh.nondegenerate_faces()
            if not valid_faces_mask.all():
                mesh.update_faces(valid_faces_mask)
            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()
            log.info("Successfully loaded via libigl: %s vertices, %s faces", len(mesh.vertices), len(mesh.faces))
            return mesh, ""
        except Exception as e2:
            log.info("Both loaders failed!")
            return None, f"Error loading mesh: {str(e)}; Fallback error: {str(e2)}"


class JarJarLoadMesh(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JarJarLoadMesh",
            display_name="JarJar: Load Mesh",
            category="JarJar/3D",
            description=(
                "Load a mesh from an absolute file path on disk. "
                "GLB/OBJ/PLY/STL/etc., SPZ splats, or splat PLYs (x,y,z + f_dc_* + opacity). "
                "Splats preserve texture as vertex_colors; with open3d installed they "
                "become watertight meshes with faces for rigging, otherwise "
                "colored point clouds. Outputs a trimesh.Trimesh."
            ),
            inputs=[
                io.String.Input(
                    "file_path",
                    default="",
                    multiline=False,
                    tooltip="Absolute path to the mesh file, e.g. C:\\path\\to\\model.glb or C:\\path\\to\\model.spz",
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
            return os.path.getmtime(file_path)
        return file_path

    @classmethod
    def execute(cls, file_path):
        file_path = (file_path or "").strip()
        if not file_path:
            raise ValueError("File path cannot be empty. Provide the full absolute path to a mesh file.")

        if (file_path.startswith('"') and file_path.endswith('"')) or \
           (file_path.startswith("'") and file_path.endswith("'")):
            file_path = file_path[1:-1].strip()

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        _mm().throw_exception_if_processing_interrupted()

        loaded_mesh, error = load_mesh_file(file_path)
        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        try:
            n_faces = len(getattr(loaded_mesh, 'faces', []))
        except Exception:
            log.debug("LoadMesh face count unreadable", exc_info=True)
            n_faces = 0
        log.info("Loaded: %s vertices, %s faces", len(loaded_mesh.vertices), n_faces)

        return io.NodeOutput(loaded_mesh)
