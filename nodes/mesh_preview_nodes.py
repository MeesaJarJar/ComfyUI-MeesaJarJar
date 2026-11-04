"""JarJarPreviewMesh - display a TRIMESH or mesh file in the 3D viewport.

Core viewers (Preview 3D, Preview Splat, Preview Point Cloud) only accept File3D,
so a raw TRIMESH from JarJarLoadMesh / JarJarLoadSplatPly / Hunyuan InPaint has
no viewer. This node bridges that: it exports the mesh to a temp file and opens
the same interactive 3D viewport core nodes use.

Two inputs (either is enough, mesh object wins if both are wired):
- mesh: TRIMESH object. Faces -> temp GLB (textured via vertex_colors).
  No faces (point-cloud fallback) -> temp PLY.
- mesh_path: path to a mesh file on disk (e.g. InPaint's output_glb_path).
  Absolute or relative to the output dir. The file is COPIED to temp (never
  modified) so PBR textures survive - no trimesh round-trip.

Temp files are managed by ComfyUI, nothing lands in output/.
"""

import logging
import math
import os
import shutil
import uuid

import numpy as np
import torch
import folder_paths
from comfy_api.latest import IO, UI

log = logging.getLogger("jarjar")


class JarJarPreviewMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarPreviewMesh",
            display_name="JarJar: Preview Mesh",
            category="JarJar/3D",
            description=(
                "Display a TRIMESH object and/or a mesh file (e.g. InPaint's "
                "output_glb_path) in the interactive 3D viewport. Mesh objects "
                "with faces preview as GLB, point clouds as PLY; files are "
                "copied to temp untouched so PBR textures survive. Passes the "
                "mesh through so it can sit mid-chain."
            ),
            is_output_node=True,
            # NOTE on input order: value-widgets MUST be declared required-first
            # (width, height, then mesh_path). The frontend creates widgets in
            # schema order while the server consumes widgets_values
            # required-first - any other order shifts values into the wrong
            # inputs (width once received mesh_path's ""). Sockets (mesh) can
            # stay wherever; keep mesh first so its link slot is stable.
            inputs=[
                IO.Custom("TRIMESH").Input(
                    "mesh",
                    optional=True,
                    tooltip="Mesh object to display.",
                ),
                IO.Load3D.Input("viewport_state", optional=True),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1,
                             tooltip="Render width of the viewport in pixels."),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1,
                             tooltip="Render height of the viewport in pixels."),
                IO.Load3DModelInfo.Input("model_3d_info", optional=True, advanced=True),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True,
                                      tooltip="Viewport camera information: position, look-at target, zoom, and type."),
                IO.String.Input(
                    "mesh_path",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Mesh file on disk (absolute, or relative to the output "
                            "folder, e.g. InPaint's output_glb_path). Copied to temp "
                            "for preview, never modified.",
                ),
            ],
            outputs=[
                IO.Custom("TRIMESH").Output(display_name="mesh"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, mesh=None, mesh_path="", viewport_state=None,
                width: int = 1024, height: int = 1024, **kwargs) -> IO.NodeOutput:
        mesh_path = (mesh_path or kwargs.get("mesh_path") or "").strip()
        # Defensive: a path STRING wired into the mesh input (easy miswire -
        # both dots sit next to each other) is treated as mesh_path.
        if isinstance(mesh, str):
            mesh_path = mesh_path or mesh.strip()
            mesh = None
        if mesh is not None:
            # Anything that is not a trimesh/Scene (e.g. an IMAGE tensor wired
            # here by mistake) gets a clear error instead of a cryptic
            # AttributeError deep in export().
            try:
                import trimesh
                _MESH_TYPES = (trimesh.Trimesh, trimesh.Scene, trimesh.points.PointCloud)
            except Exception:
                log.debug("PreviewMesh trimesh types unavailable", exc_info=True)
                _MESH_TYPES = ()
            if _MESH_TYPES and not isinstance(mesh, _MESH_TYPES):
                raise ValueError(
                    "JarJarPreviewMesh: 'mesh' input got {} - wire a TRIMESH "
                    "(e.g. InPaint trimesh) or a path STRING into mesh_path, "
                    "not an IMAGE.".format(type(mesh).__name__)
                )
        if mesh is None and not mesh_path:
            raise ValueError(
                "JarJarPreviewMesh needs a mesh object or a mesh_path string."
            )
        if mesh is not None:
            # NOTE: no `or []` guard here - `faces` is a numpy array and its
            # truth value is ambiguous (raises), so check None explicitly.
            try:
                faces = getattr(mesh, "faces", None)
                n_faces = len(faces) if faces is not None else 0
            except Exception:
                log.debug("PreviewMesh face count unreadable", exc_info=True)
                n_faces = 0
            if n_faces > 0:
                # Prefer GLB (keeps vertex colors / PBR textures). Textured
                # meshes can fail GLB export (missing image refs) - fall back
                # to OBJ, which the viewport also displays.
                last_err = None
                for ext in ("glb", "obj"):
                    filename = "jarjar_preview_{}.{}".format(uuid.uuid4().hex, ext)
                    full_path = os.path.join(folder_paths.get_temp_directory(), filename)
                    try:
                        mesh.export(full_path)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        continue
                if last_err is not None:
                    raise ValueError(
                        "JarJarPreviewMesh: could not export mesh for preview "
                        "({}).".format(last_err)
                    )
            else:
                ext = "ply"
                filename = "jarjar_preview_{}.{}".format(uuid.uuid4().hex, ext)
                full_path = os.path.join(folder_paths.get_temp_directory(), filename)
                mesh.export(full_path)
            log.info("JarJarPreviewMesh: previewing mesh object (%d faces) as %s.",
                     n_faces, filename)
        else:
            if not os.path.isabs(mesh_path):
                mesh_path = os.path.join(folder_paths.get_output_directory(), mesh_path)
            if not os.path.exists(mesh_path):
                raise ValueError("Mesh file not found: {}".format(mesh_path))
            ext = os.path.splitext(mesh_path)[1].lower().lstrip(".") or "glb"
            filename = "jarjar_preview_{}.{}".format(uuid.uuid4().hex, ext)
            full_path = os.path.join(folder_paths.get_temp_directory(), filename)
            shutil.copyfile(mesh_path, full_path)
            log.info("JarJarPreviewMesh: previewing file %s as %s.", mesh_path, filename)
            try:
                import trimesh
                loaded = trimesh.load(mesh_path, process=False)
                # Multi-geometry GLBs load as Scene - flatten like JarJarLoadMesh.
                if isinstance(loaded, trimesh.Scene):
                    mesh = loaded.dump(concatenate=True)
                else:
                    mesh = loaded
            except Exception as e:
                log.warning("JarJarPreviewMesh: passthrough load failed (%s); "
                            "preview still works, mesh output is None.", e)
                mesh = None
        viewport_state = viewport_state if isinstance(viewport_state, dict) else {}
        camera_info_input = kwargs.get("camera_info", None)
        camera_info = camera_info_input if camera_info_input is not None else viewport_state.get("camera_info")
        model_3d_info_input = kwargs.get("model_3d_info", None)
        model_3d_info = model_3d_info_input if model_3d_info_input is not None else viewport_state.get("model_3d_info", [])
        return IO.NodeOutput(
            mesh,
            model_3d_info,
            camera_info,
            width,
            height,
            ui=UI.PreviewUI3DAdvanced(filename, camera_info, model_3d_info, folder_type=IO.FolderType.temp),
        )


class JarJarFrameCamera(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarFrameCamera",
            display_name="JarJar: Frame Camera on Mesh",
            category="JarJar/3D",
            description=(
                "Build a LOAD3D_CAMERA that frames a TRIMESH at the origin of "
                "the Y-up viewer (mesh Z-up is stood upright). Wire into a "
                "3D preview's camera_info to override a lost viewport camera."
            ),
            inputs=[
                IO.Custom("TRIMESH").Input(
                    "mesh",
                    tooltip="Mesh to frame.",
                ),
                IO.Float.Input("yaw_deg", default=30.0, min=-180.0, max=180.0, step=1.0),
                IO.Float.Input("pitch_deg", default=20.0, min=-89.0, max=89.0, step=1.0),
                IO.Float.Input("distance_scale", default=2.2, min=1.1, max=10.0, step=0.1,
                               tooltip="Camera distance as a multiple of the mesh size."),
            ],
            outputs=[
                IO.Load3DCamera.Output(display_name="camera_info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, yaw_deg=30.0, pitch_deg=20.0, distance_scale=2.2) -> IO.NodeOutput:
        if mesh is None:
            raise ValueError("JarJarFrameCamera needs a mesh.")
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if verts.shape[0] == 0:
            raise ValueError("JarJarFrameCamera got an empty mesh.")
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        center = (lo + hi) / 2.0
        size = float(max(hi - lo)) or 1.0
        # Mesh is Z-up, viewer is Y-up: stand it upright for framing only.
        tgt = np.array([center[0], center[2], center[1]])
        yaw, pitch = math.radians(float(yaw_deg)), math.radians(float(pitch_deg))
        dist = size * float(distance_scale)
        eye = tgt + dist * np.array([math.sin(yaw) * math.cos(pitch),
                                     math.sin(pitch),
                                     math.cos(yaw) * math.cos(pitch)])
        z = eye - tgt
        z /= max(np.linalg.norm(z), 1e-12)
        up0 = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.999 else np.array([0.0, 0.0, 1.0])
        x = np.cross(up0, z)
        x /= max(np.linalg.norm(x), 1e-12)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)
        t = float(R[0, 0] + R[1, 1] + R[2, 2])
        if t > 0.0:
            s = 0.5 / math.sqrt(t + 1.0)
            qw, qx, qy, qz = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
        xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
        return IO.NodeOutput({
            "position": xyz(eye),
            "target": xyz(tgt),
            "quaternion": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
            "fov": 45.0,
            "cameraType": "perspective",
            "zoom": 1.0,
        })


class JarJarPreviewMeshImage(IO.ComfyNode):
    """Render shaded orthographic view(s) of a TRIMESH to IMAGE tensors."""

    _VIEWS = {
        "front": np.array([1.0, 0.0, 0.0]),
        "side": np.array([0.0, 0.0, 1.0]),
        "top": np.array([0.0, 1.0, 0.0]),
        "iso": np.array([1.0, 0.7, 1.0]),
    }
    _BASE_COLOR = np.array([0.62, 0.68, 0.78], dtype=np.float64)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarPreviewMeshImage",
            display_name="JarJar: Preview Mesh (Image)",
            category="JarJar/3D",
            description=(
                "Render shaded preview image(s) of a mesh (front / side / top / iso / all). "
                "Colored meshes keep their vertex colors; plain meshes render slate. "
                "Point clouds (0 faces) render as points. Read-only."
            ),
            inputs=[
                IO.Custom("TRIMESH").Input("mesh"),
                IO.Combo.Input(
                    "view", options=["all", "front", "side", "top", "iso"],
                    default="all",
                    tooltip="Which orthographic view(s) to render. 'all' returns 4 images (front, side, top, iso).",
                ),
                IO.Int.Input(
                    "size", default=512, min=128, max=2048, step=64,
                    tooltip="Output image height/width in pixels.",
                ),
                IO.Combo.Input(
                    "background", options=["dark", "light"], default="dark",
                    tooltip="Preview background color.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def _view_basis(cls, eye):
        eye = np.asarray(eye, dtype=np.float64)
        eye = eye / (np.linalg.norm(eye) or 1.0)
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(float(eye @ world_up)) > 0.99:
            world_up = np.array([0.0, 0.0, -1.0])
        right = np.cross(eye, world_up)
        right = right / (np.linalg.norm(right) or 1.0)
        up = np.cross(right, eye)
        return right, up / (np.linalg.norm(up) or 1.0), eye

    @classmethod
    def _face_base_colors(cls, mesh, faces):
        try:
            vc = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)
        except Exception:
            log.debug("PreviewMeshImage vertex colors unreadable, using base color", exc_info=True)
            vc = None
        if vc is not None and vc.shape[0] == len(mesh.vertices) and vc.shape[1] >= 3:
            rgb = vc[:, :3]
            if rgb.max(initial=0) > 1.0:
                rgb = rgb / 255.0
            return np.clip(rgb[np.asarray(faces)].mean(axis=1), 0.0, 1.0)
        return np.tile(cls._BASE_COLOR, (len(faces), 1))

    @classmethod
    def _mpl_fig(cls, size, bg):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ValueError(f"JarJarPreviewMeshImage needs matplotlib (pip install matplotlib): {e}")
        dpi = 100
        fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-0.58, 0.58)
        ax.set_ylim(-0.58, 0.58)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        return plt, fig, ax

    @classmethod
    def _render_view(cls, vertices, faces, face_colors, eye, size, bg):
        from matplotlib.collections import PolyCollection
        right, up, eye_n = cls._view_basis(eye)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        v = (vertices - (lo + hi) * 0.5) / (float((hi - lo).max()) or 1.0)
        px, py, depth = v @ right, v @ up, v @ eye_n
        tris = np.stack([px[faces[:, 0]], py[faces[:, 0]],
                         px[faces[:, 1]], py[faces[:, 1]],
                         px[faces[:, 2]], py[faces[:, 2]]], axis=1).reshape(len(faces), 3, 2)
        fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
        fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
        fn_len[fn_len == 0] = 1.0
        light = (eye_n + np.array([0.35, 0.45, 0.0]))
        light = light / np.linalg.norm(light)
        shade = 0.35 + 0.65 * np.clip((fn / fn_len) @ light, 0.0, 1.0)
        colors = np.clip(face_colors * shade[:, None], 0.0, 1.0)
        order = np.argsort(depth[faces].mean(axis=1))
        plt, fig, ax = cls._mpl_fig(size, bg)
        ax.add_collection(PolyCollection(tris[order], facecolors=colors[order],
                                        edgecolors="none", antialiaseds=True))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32) / 255.0
        plt.close(fig)
        return buf[:, :, :3].copy()

    @classmethod
    def _render_points(cls, vertices, point_colors, eye, size, bg):
        right, up, eye_n = cls._view_basis(eye)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        v = (vertices - (lo + hi) * 0.5) / (float((hi - lo).max()) or 1.0)
        px, py, depth = v @ right, v @ up, v @ eye_n
        order = np.argsort(depth)
        sel = order
        if len(order) > 150000:
            sel = np.random.default_rng(0).choice(len(order), 150000, replace=False)
            sel = sel[np.argsort(depth[sel])]
        plt, fig, ax = cls._mpl_fig(size, bg)
        ax.scatter(px[sel], py[sel], c=np.clip(point_colors[sel], 0.0, 1.0), s=1.0,
                   linewidths=0, antialiased=False)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32) / 255.0
        plt.close(fig)
        return buf[:, :, :3].copy()

    @classmethod
    def execute(cls, mesh, view="all", size=512, background="dark") -> IO.NodeOutput:
        if mesh is None:
            raise ValueError("No mesh provided.")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            raise ValueError("Cannot preview an empty mesh (0 vertices).")
        faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3), dtype=np.int64)))
        size = int(size or 512)
        bg = (0.09, 0.09, 0.11) if background != "light" else (0.93, 0.93, 0.94)

        names = ["front", "side", "top", "iso"] if (view or "all") == "all" else [view]
        for name in names:
            if name not in cls._VIEWS:
                raise ValueError(f"Unknown view '{name}'. Choose from: all, front, side, top, iso.")

        if len(faces) == 0:
            try:
                vc = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)
            except Exception:
                log.debug("PreviewMeshImage point colors unreadable", exc_info=True)
                vc = None
            if vc is not None and vc.shape[0] == len(vertices) and vc.shape[1] >= 3:
                pc = vc[:, :3]
                if pc.max(initial=0) > 1.0:
                    pc = pc / 255.0
                pc = np.clip(pc, 0.0, 1.0)
            else:
                pc = np.tile(cls._BASE_COLOR, (len(vertices), 1))
            renders = [cls._render_points(vertices, pc, cls._VIEWS[name], size, bg) for name in names]
            log.info("Previewed point cloud: %d points, view=%s", len(vertices), view)
        else:
            use_faces = faces
            if len(use_faces) > 100000:
                log.info("PreviewMeshImage: subsampling %d -> 100000 faces", len(use_faces))
                use_faces = use_faces[np.random.default_rng(0).choice(len(use_faces), 100000, replace=False)]
            face_colors = cls._face_base_colors(mesh, use_faces)
            renders = [cls._render_view(vertices, use_faces, face_colors, cls._VIEWS[name], size, bg) for name in names]
            log.info("Previewed mesh: %d verts %d faces, view=%s", len(vertices), len(faces), view)

        return IO.NodeOutput(torch.from_numpy(np.stack(renders, axis=0).astype(np.float32)).clamp(0.0, 1.0))


NODE_CLASS_MAPPINGS = {
    "JarJarPreviewMesh": JarJarPreviewMesh,
    "JarJarFrameCamera": JarJarFrameCamera,
    "JarJarPreviewMeshImage": JarJarPreviewMeshImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarPreviewMesh": "JarJar: Preview Mesh",
    "JarJarFrameCamera": "JarJar: Frame Camera on Mesh",
    "JarJarPreviewMeshImage": "JarJar: Preview Mesh (Image)",
}
