"""JarJarSubmit3D: submit a 3D model + prompt straight to the JarJar Foundry hopper."""

import io
import json
import logging
import os
import time
import urllib.error
import urllib.request

import folder_paths
from comfy_api.latest import IO, Types

FOUNDRY_DEFAULT_URL = "https://meesajarjar.com/foundry-api.php"
FOUNDRY_FORMATS = ("ply", "ksplat", "splat", "spz", "glb")
_MAX_PROMPT = 4000


def _encode_multipart(fields, files):
    """Minimal multipart/form-data encoder (stdlib only, no extra deps)."""
    boundary = "JarJar{:x}".format(int(time.time() * 1000) & 0xFFFFFFFF)
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


def _post_model(api_url, filename, model_bytes, prompt, timeout=600):
    body, boundary = _encode_multipart(
        {"prompt": (prompt or "")[:_MAX_PROMPT]},
        {"model": (filename, "application/octet-stream", model_bytes)},
    )
    req = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Content-Type": "multipart/form-data; boundary={}".format(boundary),
            "Content-Length": str(len(body)),
            "User-Agent": "ComfyUI-JarJarSubmit3D/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        raise RuntimeError("Upload failed: {}".format(e))


class JarJarSubmit3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSubmit3D",
            display_name="Submit 3D to JarJar Foundry",
            category="JarJar/3D",
            description=(
                "Uploads a 3D model file straight into the JarJar Foundry hopper queue "
                "(foundry-api.php), with your generation prompt saved as JSON alongside it. "
                "Accepts .ply, .ksplat, .splat, .spz and .glb — wire the same link that feeds "
                "SaveGLB, plus the prompt string."
            ),
            inputs=[
                IO.MultiType.Input(
                    IO.File3DAny.Input("model_3d"),
                    types=[
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DOBJ,
                        IO.File3DFBX,
                        IO.File3DSTL,
                        IO.File3DUSDZ,
                        IO.File3DPLY,
                        IO.File3DSPLAT,
                        IO.File3DSPZ,
                        IO.File3DKSPLAT,
                        IO.File3DSplatAny,
                        IO.File3DPointCloudAny,
                        IO.File3DAny,
                    ],
                    tooltip="Mesh or 3D file to submit: .ply, .ksplat, .splat, .spz or .glb (same link as SaveGLB).",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    placeholder="Prompt used to generate this model...",
                    tooltip="Generation prompt — saved as a simple JSON sidecar with the model in the Foundry archive.",
                ),
                IO.String.Input(
                    "api_url",
                    multiline=False,
                    default=FOUNDRY_DEFAULT_URL,
                    tooltip="Foundry ingest API endpoint.",
                ),
                IO.String.Input(
                    "filename",
                    multiline=False,
                    default="",
                    placeholder="optional_custom_name.glb",
                    tooltip="Optional file name override (keep a Foundry extension: .ply, .ksplat, .splat, .spz, .glb). Empty = auto name from prompt time.",
                ),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model_3d: Types.File3D,
        prompt: str = "",
        api_url: str = FOUNDRY_DEFAULT_URL,
        filename: str = "",
    ) -> IO.NodeOutput:
        if not isinstance(model_3d, Types.File3D):
            raise TypeError(
                "JarJarSubmit3D needs a 3D file (e.g. SplatToFile3D output), "
                "got {}".format(type(model_3d).__name__)
            )
        model_bytes = model_3d.get_bytes()
        if not model_bytes:
            raise ValueError("JarJarSubmit3D: model is empty, nothing to upload.")
        ext = (model_3d.format or "ply").lower().lstrip(".")
        if ext not in FOUNDRY_FORMATS:
            raise ValueError(
                "JarJarSubmit3D: format '.{}' is not supported by the Foundry "
                "(need one of: {}).".format(ext, ", ".join("." + f for f in FOUNDRY_FORMATS))
            )
        prompt = (prompt or "")[:_MAX_PROMPT]

        filename = (filename or "").strip()
        if filename:
            over = os.path.basename(filename).lower()
            over_ext = over.rsplit(".", 1)[-1] if "." in over else ""
            if over_ext not in FOUNDRY_FORMATS:
                raise ValueError(
                    "JarJarSubmit3D: filename override must keep a Foundry extension "
                    "({}), got '{}'.".format(
                        ", ".join("." + f for f in FOUNDRY_FORMATS), os.path.basename(filename))
                )
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in prompt[:24].strip())
            filename = "jarjar_{}{}.{}".format(stamp, "_" + safe if safe else "", ext)
        if "." not in os.path.basename(filename):
            filename += "." + ext

        status, text = _post_model(api_url, os.path.basename(filename), model_bytes, prompt)
        try:
            result = json.loads(text)
        except Exception:
            result = {"status": "error" if status != 200 else "ok", "raw": text[:500]}
        if status != 200 or result.get("status") != "ok":
            raise RuntimeError(
                "JarJarSubmit3D upload failed (HTTP {}): {}".format(
                    status, result.get("message", text[:300])
                )
            )

        item = result.get("item", {})
        logging.info(
            "JarJarSubmit3D: queued '%s' as Foundry item %s (%d bytes).",
            filename, item.get("id"), len(model_bytes),
        )

        # Local copy of the simple prompt JSON, next to the ComfyUI outputs.
        try:
            out_dir = folder_paths.get_output_directory()
            sidecar = os.path.join(out_dir, "3d", os.path.splitext(os.path.basename(filename))[0] + ".json")
            os.makedirs(os.path.dirname(sidecar), exist_ok=True)
            with open(sidecar, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "prompt": prompt,
                        "file": os.path.basename(filename),
                        "foundry_item": item,
                        "submitted_at": int(time.time()),
                    },
                    fh, indent=2,
                )
        except Exception as e:
            logging.warning("JarJarSubmit3D: could not write local prompt JSON: %s", e)

        link = api_url.rsplit("/", 1)[0] + "/foundry.html"
        summary = "Foundry item #{} '{}' ({} bytes) — watch it drop at {}".format(
            item.get("id"), os.path.basename(filename), len(model_bytes), link
        )
        return IO.NodeOutput(ui={"jarjar_result": [{"text": summary, "item": item}]})
