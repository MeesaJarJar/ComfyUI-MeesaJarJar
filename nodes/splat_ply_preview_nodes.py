"""JarJarPreviewSplatPly - display a splat PLY file in the 3D viewport.

Splat PLYs (TripoSR / TRELLIS / Hunyuan3D-2 export, SuperSplat, ...) have
no faces, so the core Preview 3D mesh viewer shows nothing useful. This
node copies the file to temp and opens the same interactive viewport the
core Preview Splat node uses (UI.Saved3DModels), so gaussians render.
"""

import logging
import os
import shutil
import uuid

import folder_paths
from comfy_api.latest import IO, UI, Types

log = logging.getLogger("jarjar")


class JarJarPreviewSplatPly(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarPreviewSplatPly",
            display_name="JarJar: Preview Splat PLY",
            category="JarJar/3D",
            description=(
                "Preview a gaussian-splat PLY file (absolute path, or relative "
                "to the output folder) in the interactive 3D splat viewport. "
                "The file is copied to temp for display, never modified. "
                "Passes the File3D through so it can sit mid-chain."
            ),
            is_output_node=True,
            # NOTE: value-widgets must stay required-first (width, height,
            # then ply_path). The frontend creates widgets in schema order
            # while the server consumes widgets_values required-first.
            inputs=[
                IO.Load3D.Input("viewport_state", optional=True),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1,
                             tooltip="Render width of the viewport in pixels."),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1,
                             tooltip="Render height of the viewport in pixels."),
                IO.Load3DModelInfo.Input("model_3d_info", optional=True, advanced=True),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True,
                                      tooltip="Viewport camera information: position, look-at target, zoom, and type."),
                IO.String.Input(
                    "ply_path",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Splat PLY on disk (absolute, or relative to the output "
                            "folder). Copied to temp for preview, never modified.",
                ),
            ],
            outputs=[
                IO.File3DSplatAny.Output(display_name="model_3d"),
                IO.Load3DModelInfo.Output(display_name="model_3d_info"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, ply_path="", viewport_state=None,
                width: int = 1024, height: int = 1024, **kwargs) -> IO.NodeOutput:
        ply_path = (ply_path or kwargs.get("ply_path") or "").strip()
        if (ply_path.startswith('"') and ply_path.endswith('"')) or \
           (ply_path.startswith("'") and ply_path.endswith("'")):
            ply_path = ply_path[1:-1].strip()
        if not ply_path:
            raise ValueError("JarJarPreviewSplatPly needs a ply_path string.")
        if not os.path.isabs(ply_path):
            ply_path = os.path.join(folder_paths.get_output_directory(), ply_path)
        if not os.path.exists(ply_path):
            raise ValueError("Splat PLY file not found: {}".format(ply_path))
        if os.path.splitext(ply_path)[1].lower() != ".ply":
            raise ValueError("JarJarPreviewSplatPly needs a .ply file, got: {}".format(ply_path))
        filename = "preview_splat_{}.ply".format(uuid.uuid4().hex)
        full_path = os.path.join(folder_paths.get_temp_directory(), filename)
        shutil.copyfile(ply_path, full_path)
        log.info("JarJarPreviewSplatPly: previewing %s as %s.", ply_path, filename)
        file_3d = Types.File3D(full_path)
        viewport_state = viewport_state if isinstance(viewport_state, dict) else {}
        camera_info_input = kwargs.get("camera_info", None)
        camera_info = camera_info_input if camera_info_input is not None else viewport_state.get("camera_info")
        model_3d_info_input = kwargs.get("model_3d_info", None)
        model_3d_info = model_3d_info_input if model_3d_info_input is not None else viewport_state.get("model_3d_info", [])
        return IO.NodeOutput(
            file_3d,
            model_3d_info,
            camera_info,
            width,
            height,
            ui=UI.Saved3DModels([UI.SavedResult(filename, "", IO.FolderType.temp)], camera_info, model_3d_info),
        )


NODE_CLASS_MAPPINGS = {
    "JarJarPreviewSplatPly": JarJarPreviewSplatPly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarPreviewSplatPly": "JarJar: Preview Splat PLY",
}
