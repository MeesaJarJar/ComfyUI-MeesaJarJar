"""JarJarSplatToFilePath - SplatToFile3D twin that ALSO saves to disk and returns paths.

ComfyUI's builtin "Create 3D File (from Splat)" (SplatToFile3D) serializes a SPLAT
to an in-memory File3D (BytesIO) - it never writes a file, so there is no path
to wire into file_path STRING inputs (JarJarLoadSplatPly, JarJarSubmit3D filename, ...).

This node runs the exact same writers (ply / ksplat / spz), returns the same
File3D object for Preview/Save nodes, AND writes the bytes to
<output>/3d/<filename> so you also get a full filepath + filename as STRINGs.
"""

import io
import logging
import os
import time

import folder_paths
from comfy_api.latest import IO, Types

log = logging.getLogger("jarjar")


class JarJarSplatToFilePath(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSplatToFilePath",
            display_name="JarJar: Splat to File + Path",
            category="JarJar/3D",
            description=(
                "Serialize a gaussian SPLAT to a File3D (same writers as Create 3D "
                "File (from Splat): ply / ksplat / spz) and save it under the output "
                "3d folder. Returns model_3d for Preview/Save nodes PLUS the full "
                "filepath and filename as STRINGs for file_path inputs. "
                "Supports one splat item per batch only (uses first)."
            ),
            inputs=[
                IO.Splat.Input("splat"),
                IO.Combo.Input(
                    "format",
                    options=["ply", "ksplat", "spz"],
                    tooltip="ply: standard 3D Gaussian Splat, full spherical harmonics. "
                            "ksplat: SplatBuffer level 0, base color only. "
                            "spz: Niantic gzip-compressed, base color only.",
                ),
                IO.String.Input(
                    "filename",
                    default="",
                    multiline=False,
                    placeholder="optional_name.ply",
                    tooltip="Optional file name. Extension is forced to match format. "
                            "Empty = auto splat_YYYYMMDD_HHMMSS.<ext>.",
                ),
            ],
            outputs=[
                IO.File3DSplatAny.Output(display_name="model_3d"),
                IO.String.Output(display_name="filepath"),
                IO.String.Output(display_name="filename"),
            ],
        )

    @classmethod
    def execute(cls, splat, format="ply", filename="") -> IO.NodeOutput:
        from comfy_extras.nodes_gaussian_splat import SplatToFile3D, _real_len

        writer = SplatToFile3D.FORMAT_WRITERS.get(format)
        if writer is None:
            raise ValueError("Unsupported splat format: {!r} (need ply/ksplat/spz).".format(format))

        if splat.positions.shape[0] > 1:
            log.warning(
                "JarJarSplatToFilePath supports one item per batch only. Got %d; using first.",
                splat.positions.shape[0],
            )
        end = _real_len(splat, 0)
        data = writer(splat.positions[0, :end], splat.scales[0, :end],
                      splat.rotations[0, :end], splat.opacities[0, :end], splat.sh[0, :end])

        filename = (filename or "").strip()
        filename = os.path.basename(filename)
        if filename:
            stem = filename.rsplit(".", 1)[0] if "." in filename else filename
            filename = "{}.{}".format(stem, format)
        else:
            filename = "splat_{}.{}".format(time.strftime("%Y%m%d_%H%M%S"), format)

        out_dir = os.path.join(folder_paths.get_output_directory(), "3d")
        os.makedirs(out_dir, exist_ok=True)
        full_path = os.path.join(out_dir, filename)
        with open(full_path, "wb") as fh:
            fh.write(data)
        log.info("JarJarSplatToFilePath: wrote %s (%d bytes).", full_path, len(data))

        return IO.NodeOutput(
            Types.File3D(io.BytesIO(data), file_format=format),
            full_path,
            filename,
        )


NODE_CLASS_MAPPINGS = {
    "JarJarSplatToFilePath": JarJarSplatToFilePath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarSplatToFilePath": "JarJar: Splat to File + Path",
}
