"""JarJarOrientMesh - Rigidly re-orient a mesh (rotate about X/Y/Z in degrees)."""

import logging
import numpy as np
from scipy.spatial.transform import Rotation
from comfy_api.latest import io

log = logging.getLogger("jarjar")


def _mm():
    import comfy.model_management
    return comfy.model_management


class JarJarOrientMesh(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JarJarOrientMesh",
            display_name="JarJar: Orient Mesh",
            category="JarJar/3D",
            description=(
                "Rigidly rotate a mesh about X/Y/Z axes (degrees). "
                "Use between a Load Mesh node and rigging/preview to correct "
                "orientation (e.g. rotate_y=90 turns a character facing -X to face +Z). "
                "Faces, UVs and vertex colors are preserved; vertex normals are rotated too."
            ),
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Float.Input(
                    "rotate_x", default=0.0, min=-360.0, max=360.0, step=1.0,
                    tooltip="Rotation about the X axis in degrees (right-hand rule).",
                ),
                io.Float.Input(
                    "rotate_y", default=0.0, min=-360.0, max=360.0, step=1.0,
                    tooltip="Rotation about the Y axis in degrees (right-hand rule).",
                ),
                io.Float.Input(
                    "rotate_z", default=0.0, min=-360.0, max=360.0, step=1.0,
                    tooltip="Rotation about the Z axis in degrees (right-hand rule).",
                ),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(cls, mesh, rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
        if mesh is None:
            raise ValueError("No mesh provided.")

        rx, ry, rz = float(rotate_x or 0.0), float(rotate_y or 0.0), float(rotate_z or 0.0)
        if rx == 0.0 and ry == 0.0 and rz == 0.0:
            return io.NodeOutput(mesh)

        _mm().throw_exception_if_processing_interrupted()
        mesh = mesh.copy()

        R = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        verts = np.asarray(mesh.vertices, dtype=np.float64) @ R.T
        mesh.vertices = verts

        try:
            normals = np.asarray(mesh.vertex_normals)
            if normals.shape == verts.shape:
                mesh.vertex_normals = normals @ R.T
        except Exception:
            pass

        try:
            meta = dict(getattr(mesh, "metadata", {}) or {})
            meta["jarjar_orientation_deg"] = [rx, ry, rz]
            mesh.metadata = meta
        except Exception:
            pass

        log.info("Oriented mesh: rotate_x=%.1f rotate_y=%.1f rotate_z=%.1f", rx, ry, rz)
        return io.NodeOutput(mesh)


NODE_CLASS_MAPPINGS = {
    "JarJarOrientMesh": JarJarOrientMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarOrientMesh": "JarJar: Orient Mesh",
}
