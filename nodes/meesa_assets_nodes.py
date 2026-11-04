"""MeesaAssets ComfyUI nodes v1 — push/pull Images + 3D to the PHP-only MeesaAssets API.

Target: MeesaAssets/api.php (no auth in v1).
- MeesaAssetsPushImage  (old-style): IMAGE tensor -> PNG upload, returns url + file.
- MeesaAssetsPushModel  (old-style): STRING file_path on disk -> uploads .ply/.obj/.glb/.splat/etc.
- MeesaAssetsPullImage  (old-style): file name + api_url -> IMAGE tensor (for retrieve/edit loops).
- MeesaAssetsPushModel3D (new API, optional): File3D passthrough like JarJarSubmit3D.

Install: copy this file to ComfyUI/custom_nodes/ComfyUI-MeesaJarJar/nodes/meesa_assets_nodes.py
and register the mappings in nodes/__init__.py (see comfyui-nodes/install.py in this repo).
Stdlib-only HTTP (urllib) + torch/PIL/numpy already required by ComfyUI.
"""

import io
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("jarjar")

MEESA_DEFAULT_URL = "http://localhost:8000/api.php"
_MEESA_MAX_PROMPT = 4000


def _encode_multipart(fields, files):
    boundary = "Meesa{:x}".format(int(time.time() * 1000) & 0xFFFFFFFF)
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write("--{}\r\n".format(boundary).encode())
        buf.write('Content-Disposition: form-data; name="{}"\r\n\r\n'.format(name).encode())
        buf.write(str(value).encode("utf-8", "replace"))
        buf.write(b"\r\n")
    for name, (filename, content_type, data) in files.items():
        buf.write("--{}\r\n".format(boundary).encode())
        buf.write('Content-Disposition: form-data; name="{}"; filename="{}"\r\n'.format(name, filename).encode())
        buf.write("Content-Type: {}\r\n\r\n".format(content_type).encode())
        buf.write(data)
        buf.write(b"\r\n")
    buf.write("--{}--\r\n".format(boundary).encode())
    return buf.getvalue(), boundary


def _post_file(api_url, field, filename, data_bytes, prompt="", title="", tags="", timeout=600):
    body, boundary = _encode_multipart(
        {"prompt": (prompt or "")[:_MEESA_MAX_PROMPT], "title": title or "", "tags": tags or ""},
        {field: (filename, "application/octet-stream", data_bytes)},
    )
    req = urllib.request.Request(
        api_url, data=body,
        headers={"Content-Type": "multipart/form-data; boundary={}".format(boundary),
                 "Content-Length": str(len(body)),
                 "User-Agent": "ComfyUI-MeesaAssets/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        raise RuntimeError("MeesaAssets upload failed: {}".format(e))


def _keyed(api_url, api_key):
    """Append ?key= for personal libraries. Empty key = legacy shared library."""
    api_key = (api_key or "").strip()
    if not api_key:
        return api_url
    sep = "&" if "?" in api_url else "?"
    return api_url + sep + "key=" + urllib.parse.quote(api_key)


def _check_ok(status, text):
    try:
        result = json.loads(text)
    except Exception:
        result = {"status": "error" if status != 200 else "ok", "raw": text[:500]}
    if status != 200 or result.get("status") != "ok":
        raise RuntimeError("MeesaAssets upload failed (HTTP {}): {}".format(status, result.get("message", text[:300])))
    return result


class JarJarPushImage:
    """Upload an IMAGE batch (first frame) as PNG to MeesaAssets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "api_url": ("STRING", {"default": MEESA_DEFAULT_URL, "multiline": False}),
            "title": ("STRING", {"default": "", "multiline": False}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "tags": ("STRING", {"default": "", "multiline": False}),
            "filename": ("STRING", {"default": "", "multiline": False}),
        },
            "optional": {
            "api_key": ("STRING", {"default": "", "multiline": False}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "file")
    FUNCTION = "push"
    CATEGORY = "JarJar/Meesa"
    DESCRIPTION = "Pushes the first IMAGE in the batch to MeesaAssets api.php as PNG with title/prompt/tags. api_key empty = shared library, else your personal library. Returns download url + stored file name."
    OUTPUT_NODE = True

    def push(self, image, api_url=MEESA_DEFAULT_URL, title="", prompt="", tags="", filename="", api_key=""):
        if isinstance(image, list):
            image = image[0]
        arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.array(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        data = buf.getvalue()

        filename = (filename or "").strip() or "meesa_{}.png".format(time.strftime("%Y%m%d_%H%M%S"))
        filename = os.path.basename(filename)
        if not filename.lower().endswith(".png"):
            filename += ".png"
        endpoint = _keyed(api_url or MEESA_DEFAULT_URL, api_key)
        status, text = _post_file(endpoint, "file", filename, data, prompt, title, tags)
        result = _check_ok(status, text)
        item = result.get("item", {})
        base = (api_url or MEESA_DEFAULT_URL).rsplit("/", 1)[0]
        url = base + "/" + item.get("url", "api.php?action=download&file=" + urllib.parse.quote(item.get("file", filename)))
        url = _keyed(url, api_key)
        return (url, item.get("file", filename))


class JarJarPushModel:
    """Upload a model file from disk (STRING path) to MeesaAssets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "file_path": ("STRING", {"default": "", "multiline": False}),
            "api_url": ("STRING", {"default": MEESA_DEFAULT_URL, "multiline": False}),
            "title": ("STRING", {"default": "", "multiline": False}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "tags": ("STRING", {"default": "", "multiline": False}),
        },
            "optional": {
            "api_key": ("STRING", {"default": "", "multiline": False}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "file")
    FUNCTION = "push"
    CATEGORY = "JarJar/Meesa"
    DESCRIPTION = "Uploads a 3D model file from disk (.ply .obj .glb .gltf .stl .splat .ksplat .spz, audio/video too) to MeesaAssets. Wire a SaveGLB path or any file path. Returns url + stored file name."
    OUTPUT_NODE = True

    def push(self, file_path, api_url=MEESA_DEFAULT_URL, title="", prompt="", tags="", api_key=""):
        file_path = (file_path or "").strip().strip('"')
        if not file_path or not os.path.isfile(file_path):
            raise ValueError("MeesaAssetsPushModel: file not found: {!r}".format(file_path))
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext not in ("ply", "obj", "glb", "gltf", "stl", "splat", "ksplat", "spz",
                       "png", "jpg", "jpeg", "webp", "gif", "bmp",
                       "mp3", "wav", "ogg", "flac", "mp4", "webm", "mov", "m4v"):
            raise ValueError("MeesaAssetsPushModel: .{} not accepted (need image, 3D, audio or video).".format(ext))
        with open(file_path, "rb") as fh:
            data = fh.read()
        if not data:
            raise ValueError("MeesaAssetsPushModel: file is empty.")
        endpoint = _keyed(api_url or MEESA_DEFAULT_URL, api_key)
        status, text = _post_file(endpoint, "file", os.path.basename(file_path), data, prompt, title, tags)
        result = _check_ok(status, text)
        item = result.get("item", {})
        base = (api_url or MEESA_DEFAULT_URL).rsplit("/", 1)[0]
        url = _keyed(base + "/" + item.get("url", ""), api_key)
        return (url, item.get("file", os.path.basename(file_path)))


class JarJarPullImage:
    """Download an image asset by file name from MeesaAssets -> IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "file": ("STRING", {"default": "", "multiline": False}),
            "api_url": ("STRING", {"default": MEESA_DEFAULT_URL, "multiline": False}),
        },
            "optional": {
            "api_key": ("STRING", {"default": "", "multiline": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pull"
    CATEGORY = "JarJar/Meesa"
    DESCRIPTION = "Retrieves an image from MeesaAssets by stored file name (see portfolio or api.php?action=list). api_key unlocks your private files. Returns IMAGE tensor for edit loops."

    def pull(self, file, api_url=MEESA_DEFAULT_URL, api_key=""):
        file = (file or "").strip()
        if not file:
            raise ValueError("MeesaAssetsPullImage: give a stored file name, e.g. 1712345678_sword.png")
        base = (api_url or MEESA_DEFAULT_URL).rsplit("/", 1)[0] if (api_url or "").endswith(".php") else (api_url or MEESA_DEFAULT_URL).rstrip("/")
        endpoint = (api_url or MEESA_DEFAULT_URL) if (api_url or "").endswith(".php") else base + "/api.php"
        url = _keyed(endpoint + "?action=download&file=" + urllib.parse.quote(os.path.basename(file)), api_key)
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError("MeesaAssetsPullImage: download failed (HTTP {}): {}".format(e.code, e.read().decode('utf-8', 'replace')[:300]))
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise RuntimeError("MeesaAssetsPullImage: not an image ({}). 3D files cannot come back as IMAGE — use the file URL directly.".format(e))
        arr = np.array(img).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "JarJarPushImage": JarJarPushImage,
    "JarJarPushModel": JarJarPushModel,
    "JarJarPullImage": JarJarPullImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarPushImage": "JarJar Push Image",
    "JarJarPushModel": "JarJar Push Model",
    "JarJarPullImage": "JarJar Pull Image",
}

# ---- File3D node: always registered (deterministic key) ----
try:
    from comfy_api.latest import IO, Types  # noqa: E402

    _HAS_COMFY_API = True
except Exception as _e:
    IO = None  # type: ignore
    Types = None  # type: ignore
    _HAS_COMFY_API = False
    log.debug("comfy_api unavailable, JarJarPushModel3D stub registered: %s", _e, exc_info=True)

MEESA_3D_FORMATS = ("ply", "obj", "glb", "gltf", "stl", "splat", "ksplat", "spz")


if _HAS_COMFY_API:
    class JarJarPushModel3D(IO.ComfyNode):
        @classmethod
        def define_schema(cls):
            return IO.Schema(
                node_id="JarJarPushModel3D",
                display_name="JarJar Push Model 3D",
                category="JarJar/Meesa",
                description="Uploads a File3D link (same wire as SaveGLB / SplatToFile3D) straight to MeesaAssets api.php.",
                inputs=[
                    IO.MultiType.Input(
                        IO.File3DAny.Input("model_3d"),
                        types=[IO.File3DGLB, IO.File3DGLTF, IO.File3DOBJ, IO.File3DFBX, IO.File3DSTL,
                               IO.File3DUSDZ, IO.File3DPLY, IO.File3DSPLAT, IO.File3DSPZ, IO.File3DKSPLAT,
                               IO.File3DSplatAny, IO.File3DPointCloudAny, IO.File3DAny],
                        tooltip="Mesh or splat file to push to your MeesaAssets portfolio.",
                    ),
                    IO.String.Input("prompt", multiline=True, default="", placeholder="Prompt…"),
                    IO.String.Input("title", multiline=False, default="", placeholder="Title…"),
                    IO.String.Input("tags", multiline=False, default="", placeholder="tag1, tag2"),
                    IO.String.Input("api_url", multiline=False, default=MEESA_DEFAULT_URL),
                    IO.String.Input("filename", multiline=False, default="", placeholder="optional_custom_name.glb"),
                    IO.String.Input("api_key", multiline=False, default="", placeholder="API key for your library (empty = shared)", optional=True),
                ],
                outputs=[IO.String.Output("url"), IO.String.Output("file")],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, model_3d: Types.File3D, prompt: str = "", title: str = "", tags: str = "",
                    api_url: str = MEESA_DEFAULT_URL, filename: str = "", api_key: str = "") -> IO.NodeOutput:
            if not isinstance(model_3d, Types.File3D):
                raise TypeError("MeesaAssetsPushModel3D needs a 3D file, got {}".format(type(model_3d).__name__))
            data = model_3d.get_bytes()
            if not data:
                raise ValueError("MeesaAssetsPushModel3D: model is empty.")
            ext = (model_3d.format or "glb").lower().lstrip(".")
            if ext not in MEESA_3D_FORMATS and ext != "fbx" and ext != "usdz":
                raise ValueError("MeesaAssetsPushModel3D: .{} not accepted in v1.".format(ext))
            filename = (filename or "").strip() or "meesa_{}.{}".format(time.strftime("%Y%m%d_%H%M%S"), ext)
            filename = os.path.basename(filename)
            endpoint = _keyed(api_url or MEESA_DEFAULT_URL, api_key)
            status, text = _post_file(endpoint, "file", filename, data, prompt, title, tags)
            result = _check_ok(status, text)
            item = result.get("item", {})
            base = (api_url or MEESA_DEFAULT_URL).rsplit("/", 1)[0]
            url = _keyed(base + "/" + item.get("url", ""), api_key)
            return IO.NodeOutput(url, item.get("file", filename))

else:
    class JarJarPushModel3D:
        """Stub when comfy_api File3D is unavailable. Same registry key, clean error."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "model_3d": ("STRING", {"default": "", "multiline": False}),
                "api_url": ("STRING", {"default": MEESA_DEFAULT_URL, "multiline": False}),
            },
                "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "title": ("STRING", {"default": "", "multiline": False}),
                "tags": ("STRING", {"default": "", "multiline": False}),
                "filename": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }}

        RETURN_TYPES = ("STRING", "STRING")
        RETURN_NAMES = ("url", "file")
        FUNCTION = "execute"
        CATEGORY = "JarJar/Meesa"
        DESCRIPTION = "Requires comfy_api File3D (update ComfyUI). Stub keeps registry deterministic."
        OUTPUT_NODE = True

        def execute(self, model_3d="", api_url=MEESA_DEFAULT_URL, prompt="", title="", tags="", filename="", api_key=""):
            raise RuntimeError("comfy_api required: JarJarPushModel3D needs ComfyUI with comfy_api File3D support.")


NODE_CLASS_MAPPINGS["JarJarPushModel3D"] = JarJarPushModel3D
NODE_DISPLAY_NAME_MAPPINGS["JarJarPushModel3D"] = "JarJar Push Model 3D"
