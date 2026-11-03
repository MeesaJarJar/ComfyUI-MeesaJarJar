"""JarJar splat retexture - new texture, same shape.

A gaussian-splat PLY stores shape (x,y,z,scale_*,rot_*,opacity) separately
from color (f_dc_0/1/2 + optional f_rest_*). Retexture keeps every shape
field untouched and only rewrites the SH color fields, so geometry,
silhouette and density are identical.

Three nodes, one per strategy:
- Anchor: project one sharp front image onto the visible gaussians.
- Multiview: blend N enhanced orbit views (uniform yaw orbit).
- Bake: copy vertex_colors from a painted/unwrapped mesh onto the gaussians.
"""

import io
import logging
import os
import time

import folder_paths
import numpy as np
from comfy_api.latest import IO, Types
from plyfile import PlyData
from scipy.spatial import cKDTree

log = logging.getLogger("jarjar")

_SH_C0 = 0.28209479177387814


def _to_numpy_image(image):
    arr = np.asarray(image.cpu().numpy() if hasattr(image, "cpu") else image)
    if arr.ndim == 4:
        return np.clip(arr.astype(np.float64), 0.0, 1.0)
    return np.clip(arr[None].astype(np.float64), 0.0, 1.0)


def _load_splat(path):
    ply = PlyData.read(path)
    names = list(ply["vertex"].data.dtype.names)
    for req in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"):
        if req not in names:
            raise ValueError("Not a gaussian-splat PLY: missing '{}'".format(req))
    xyz = np.stack([np.asarray(ply["vertex"].data["x"], dtype=np.float64),
                    np.asarray(ply["vertex"].data["y"], dtype=np.float64),
                    np.asarray(ply["vertex"].data["z"], dtype=np.float64)], axis=1)
    nml = None
    if {"nx", "ny", "nz"} <= set(names):
        nml = np.stack([np.asarray(ply["vertex"].data["nx"], dtype=np.float64),
                        np.asarray(ply["vertex"].data["ny"], dtype=np.float64),
                        np.asarray(ply["vertex"].data["nz"], dtype=np.float64)], axis=1)
        n = np.linalg.norm(nml, axis=1, keepdims=True)
        nml = np.where(n > 1e-12, nml / np.maximum(n, 1e-12), 0.0)
    return ply, xyz, nml


def _camera(yaw_deg, pitch_deg, center, radius, fov_deg):
    yaw, pitch = np.radians([yaw_deg, pitch_deg])
    dist = float(radius * 2.0 / max(np.tan(np.radians(max(fov_deg, 5.0)) / 2.0), 1e-6))
    pos = center + dist * np.array([np.cos(pitch) * np.sin(yaw),
                                    -np.cos(pitch) * np.cos(yaw),
                                    np.sin(pitch)])
    fwd = (center - pos) / np.linalg.norm(center - pos)
    up = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, np.stack([right, up, -fwd], axis=0)


def _project(xyz, img_w, img_h, yaw_deg, pitch_deg, fov_deg):
    center = np.mean(xyz, axis=0)
    radius = float(np.median(np.linalg.norm(xyz - center, axis=1)) + 1e-6)
    pos, R = _camera(yaw_deg, pitch_deg, center, radius, fov_deg)
    cam = (xyz - pos) @ R.T
    depth = -cam[:, 2]
    focal = 0.5 * img_w / max(np.tan(np.radians(max(fov_deg, 5.0)) / 2.0), 1e-6)
    u = cam[:, 0] / np.maximum(depth, 1e-6) * focal + img_w * 0.5
    v = -cam[:, 1] / np.maximum(depth, 1e-6) * focal + img_h * 0.5
    view_dir = (xyz - pos) / np.maximum(np.linalg.norm(xyz - pos, axis=1, keepdims=True), 1e-12)
    return u, v, depth, view_dir, pos


def _bilinear(img, u, v):
    h, w, _ = img.shape
    x = np.clip(u, 0, w - 1.001)
    y = np.clip(v, 0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
            (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def _visible_winners(u, v, depth, img_w, img_h):
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (depth > 0)
    idx = np.nonzero(inside)[0]
    if idx.size == 0:
        return idx
    pix = np.floor(v[idx]).astype(np.int64) * img_w + np.floor(u[idx]).astype(np.int64)
    order = np.argsort(depth[idx], kind="stable")
    _, first = np.unique(pix[order], return_index=True)
    return idx[order[first]]


def _save_recolored(ply, fdc_new, filename):
    v = ply["vertex"].data
    v["f_dc_0"] = fdc_new[:, 0].astype(np.float32)
    v["f_dc_1"] = fdc_new[:, 1].astype(np.float32)
    v["f_dc_2"] = fdc_new[:, 2].astype(np.float32)
    filename = os.path.basename((filename or "").strip()) or "splat_retex_{}.ply".format(
        time.strftime("%Y%m%d_%H%M%S"))
    if not filename.lower().endswith(".ply"):
        filename = filename.rsplit(".", 1)[0] + ".ply"
    out_dir = os.path.join(folder_paths.get_output_directory(), "3d")
    os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, filename)
    ply.write(full)
    with open(full, "rb") as fh:
        file3d = Types.File3D(io.BytesIO(fh.read()), file_format="ply")
    log.info("JarJar retexture: wrote %s", full)
    return full, filename, file3d


def _fdc_from_rgb(rgb):
    return (np.clip(rgb, 0.0, 1.0) - 0.5) / _SH_C0


def _rgb_from_ply(ply):
    v = ply["vertex"].data
    return np.stack([np.asarray(v["f_dc_0"], dtype=np.float64),
                     np.asarray(v["f_dc_1"], dtype=np.float64),
                     np.asarray(v["f_dc_2"], dtype=np.float64)], axis=1) * _SH_C0 + 0.5


class JarJarSplatAnchorTexture(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSplatAnchorTexture",
            display_name="JarJar: Splat Anchor Texture (front)",
            category="JarJar/3D",
            description=(
                "Retexture a splat PLY with one sharp front image, same shape. "
                "Only visible gaussians (depth-tested) get new f_dc color, "
                "weighted by normal facing + edge feather. Writes a new PLY."
            ),
            inputs=[
                IO.String.Input("file_path", default="", multiline=False,
                                tooltip="Absolute path to the splat PLY."),
                IO.Image.Input("image",
                               tooltip="Single sharp reference (e.g. upscaled Flux front). Uses first of batch."),
                IO.Float.Input("yaw_deg", default=0.0, min=-180.0, max=180.0, step=1.0),
                IO.Float.Input("pitch_deg", default=15.0, min=-90.0, max=90.0, step=1.0),
                IO.Float.Input("fov_deg", default=40.0, min=5.0, max=90.0, step=1.0),
                IO.Float.Input("edge_feather", default=0.15, min=0.0, max=0.49, step=0.01,
                               tooltip="Ignore gaussians this close to the image border (0-0.49)."),
                IO.Float.Input("recolor_strength", default=1.0, min=0.0, max=1.0, step=0.05),
                IO.String.Input("filename", default="", multiline=False, optional=True),
            ],
            outputs=[
                IO.String.Output(display_name="filepath"),
                IO.String.Output(display_name="filename"),
                IO.File3DSplatAny.Output(display_name="model_3d"),
            ],
        )

    @classmethod
    def execute(cls, file_path, image, yaw_deg=0.0, pitch_deg=15.0, fov_deg=40.0,
                edge_feather=0.15, recolor_strength=1.0, filename="") -> IO.NodeOutput:
        file_path = (file_path or "").strip()
        if not os.path.exists(file_path):
            raise ValueError("Splat file not found: {}".format(file_path))
        ref = _to_numpy_image(image)[0]
        ih, iw, _ = ref.shape
        ply, xyz, nml = _load_splat(file_path)
        u, v, depth, view_dir, _ = _project(xyz, iw, ih, float(yaw_deg), float(pitch_deg), float(fov_deg))
        m = float(edge_feather)
        u, v = u / iw, v / ih
        border = (u > m) & (u < 1 - m) & (v > m) & (v < 1 - m)
        u, v = u * iw, v * ih
        win = _visible_winners(u, v, depth, iw, ih)
        win = win[border[win]]
        if win.size == 0:
            raise ValueError("No gaussians visible from yaw={} pitch={} - check camera.".format(yaw_deg, pitch_deg))
        sample = _bilinear(ref, u[win], v[win])
        if nml is not None:
            facing = np.clip(np.einsum("ij,ij->i", nml[win], -view_dir[win]), 0.0, 1.0)
        else:
            facing = np.ones(win.shape)
        w = (facing * float(recolor_strength))[:, None]
        base = np.clip(_rgb_from_ply(ply), 0.0, 1.0)
        base[win] = base[win] * (1 - w) + sample * w
        full, name, file3d = _save_recolored(ply, _fdc_from_rgb(base), filename or "")
        return IO.NodeOutput(full, name, file3d)


class JarJarSplatMultiviewTexture(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSplatMultiviewTexture",
            display_name="JarJar: Splat Multiview Texture (orbit)",
            category="JarJar/3D",
            description=(
                "Retexture a splat PLY with N enhanced orbit views, same shape. "
                "Feed a batch of enhanced renders (uniform yaw orbit); each view "
                "votes depth-tested color, averaged per gaussian. Writes a new PLY."
            ),
            inputs=[
                IO.String.Input("file_path", default="", multiline=False,
                                tooltip="Absolute path to the splat PLY."),
                IO.Image.Input("images",
                               tooltip="Batch of N enhanced orbit renders, uniform yaw step."),
                IO.Float.Input("yaw_start_deg", default=0.0, min=-180.0, max=180.0, step=1.0),
                IO.Float.Input("pitch_deg", default=15.0, min=-90.0, max=90.0, step=1.0),
                IO.Float.Input("fov_deg", default=40.0, min=5.0, max=90.0, step=1.0),
                IO.Float.Input("recolor_strength", default=1.0, min=0.0, max=1.0, step=0.05),
                IO.String.Input("filename", default="", multiline=False, optional=True),
            ],
            outputs=[
                IO.String.Output(display_name="filepath"),
                IO.String.Output(display_name="filename"),
                IO.File3DSplatAny.Output(display_name="model_3d"),
            ],
        )

    @classmethod
    def execute(cls, file_path, images, yaw_start_deg=0.0, pitch_deg=15.0,
                fov_deg=40.0, recolor_strength=1.0, filename="") -> IO.NodeOutput:
        file_path = (file_path or "").strip()
        if not os.path.exists(file_path):
            raise ValueError("Splat file not found: {}".format(file_path))
        batch = _to_numpy_image(images)
        n = batch.shape[0]
        if n == 0:
            raise ValueError("No images in batch.")
        ply, xyz, nml = _load_splat(file_path)
        acc = np.zeros_like(xyz)
        cnt = np.zeros(xyz.shape[0])
        for i in range(n):
            img = batch[i]
            ih, iw, _ = img.shape
            yaw = float(yaw_start_deg) + 360.0 * i / n
            u, v, depth, view_dir, _ = _project(xyz, iw, ih, yaw, float(pitch_deg), float(fov_deg))
            win = _visible_winners(u, v, depth, iw, ih)
            if win.size == 0:
                continue
            sample = _bilinear(img, u[win], v[win])
            if nml is not None:
                facing = np.clip(np.einsum("ij,ij->i", nml[win], -view_dir[win]), 0.0, 1.0)
            else:
                facing = np.ones(win.shape)
            acc[win] += sample * facing[:, None]
            cnt[win] += facing
        hit = cnt > 1e-6
        if not hit.any():
            raise ValueError("No gaussians hit by {} views - check pitch/fov.".format(n))
        base = np.clip(_rgb_from_ply(ply), 0.0, 1.0)
        vote = np.zeros_like(base)
        vote[hit] = acc[hit] / cnt[hit][:, None]
        s = float(recolor_strength)
        base[hit] = base[hit] * (1 - s) + np.clip(vote[hit], 0.0, 1.0) * s
        full, name, file3d = _save_recolored(ply, _fdc_from_rgb(base), filename or "")
        return IO.NodeOutput(full, name, file3d)


class JarJarSplatBakeVertexTexture(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSplatBakeVertexTexture",
            display_name="JarJar: Splat Bake Mesh Texture",
            category="JarJar/3D",
            description=(
                "Retexture a splat PLY from a mesh's vertex_colors, same shape. "
                "Each gaussian copies the nearest mesh vertex color "
                "(Poisson mesh / Hunyuan paint path). Writes a new PLY."
            ),
            inputs=[
                IO.String.Input("file_path", default="", multiline=False,
                                tooltip="Absolute path to the splat PLY."),
                IO.Custom("TRIMESH").Input("mesh", tooltip="Mesh with vertex_colors."),
                IO.Float.Input("recolor_strength", default=1.0, min=0.0, max=1.0, step=0.05),
                IO.String.Input("filename", default="", multiline=False, optional=True),
            ],
            outputs=[
                IO.String.Output(display_name="filepath"),
                IO.String.Output(display_name="filename"),
                IO.File3DSplatAny.Output(display_name="model_3d"),
            ],
        )

    @classmethod
    def execute(cls, file_path, mesh, recolor_strength=1.0, filename="") -> IO.NodeOutput:
        file_path = (file_path or "").strip()
        if not os.path.exists(file_path):
            raise ValueError("Splat file not found: {}".format(file_path))
        if mesh is None:
            raise ValueError("No mesh wired in.")
        vis = getattr(mesh, "visual", None)
        vc = None if vis is None else np.asarray(vis.vertex_colors)
        if vc is None or vc.size == 0:
            raise ValueError("Mesh has no vertex_colors to bake from.")
        vc = np.asarray(vc, dtype=np.float64)
        if vc.shape[1] == 4:
            vc = vc[:, :3] / 255.0
        elif vc.max() > 1.0:
            vc = vc / 255.0
        mv = np.asarray(mesh.vertices, dtype=np.float64)
        ply, xyz, _ = _load_splat(file_path)
        _, nn = cKDTree(mv).query(np.asarray(xyz, dtype=np.float64), k=1)
        base = np.clip(_rgb_from_ply(ply), 0.0, 1.0)
        s = float(recolor_strength)
        base = base * (1 - s) + np.clip(vc[nn][:, :3], 0.0, 1.0) * s
        full, name, file3d = _save_recolored(ply, _fdc_from_rgb(base), filename or "")
        return IO.NodeOutput(full, name, file3d)


NODE_CLASS_MAPPINGS = {
    "JarJarSplatAnchorTexture": JarJarSplatAnchorTexture,
    "JarJarSplatMultiviewTexture": JarJarSplatMultiviewTexture,
    "JarJarSplatBakeVertexTexture": JarJarSplatBakeVertexTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarSplatAnchorTexture": "JarJar: Splat Anchor Texture (front)",
    "JarJarSplatMultiviewTexture": "JarJar: Splat Multiview Texture (orbit)",
    "JarJarSplatBakeVertexTexture": "JarJar: Splat Bake Mesh Texture",
}
