"""MeesaJarJar 3D nodes - extracted from Old pack:
- JarJarSaveGLB.py (was NOT wired in Old __init__.py, now included)
- JarJarUVUnwrap.py (was NOT wired in Old __init__.py, now included)
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("jarjar")


def _resolve_blender_path(explicit=""):
    # Blender 4.4+ tested. Resolve at runtime, never hardcode an install path.
    explicit = (explicit or "").strip()
    if explicit:
        return explicit
    env_path = os.environ.get("BLENDER_PATH", "").strip()
    if env_path:
        return env_path
    return shutil.which("blender") or ""


class JarJarSaveGLB:
    """Save MESH (trimesh) to .glb file path."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {}),
                "file_path": ("STRING", {"default": "./output/mesh.glb", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "save_glb"
    CATEGORY = "JarJar/3D"
    DESCRIPTION = "Exports trimesh MESH to file_path as GLB (creates dirs, forces .glb). Returns status STRING."

    def save_glb(self, mesh, file_path: str):
        try:
            save_path = Path(file_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.suffix.lower() != '.glb':
                save_path = save_path.with_suffix('.glb')

            mesh.export(str(save_path), file_type='glb')

            success_message = f"Mesh saved successfully to: {save_path}"
            log.info("%s", success_message)

            return (success_message,)

        except (AttributeError, TypeError):
            error_msg = "Error: Input is not a valid mesh object. Expected trimesh output."
            log.warning("%s", error_msg)
            return (error_msg,)
        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Error saving mesh: {str(e)}"
            log.warning("%s", error_msg)
            return (error_msg,)
        except Exception as e:
            error_msg = f"Error saving mesh: {str(e)}"
            log.exception("Error saving mesh")
            return (error_msg,)


class JarJarUVUnwrap:
    """UV-unwrap TRIMESH via Blender Smart UV Project (background mode)."""

    def __init__(self):
        # Blender 4.4+ tested; resolved at runtime via _resolve_blender_path().
        self.blender_path = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {}),
                "output_folder": ("STRING", {"default": "./output", "multiline": False}),
                "filename": ("STRING", {"default": "unwrapped_mesh.glb", "multiline": False}),
            },
            "optional": {
                "blender_path": ("STRING", {"default": "", "multiline": False}),
                "angle_limit": ("FLOAT", {"default": 66.0, "min": 1.0, "max": 89.0, "step": 1.0}),
                "island_margin": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file_path",)
    FUNCTION = "uv_unwrap"
    CATEGORY = "JarJar/3D"
    DESCRIPTION = "Runs Blender background Smart UV Project on TRIMESH, saves unwrapped GLB. Returns output path STRING."

    def uv_unwrap(self, trimesh, output_folder: str, filename: str,
                  blender_path: str = None, angle_limit: float = 66.0,
                  island_margin: float = 0.02):
        try:
            resolved = _resolve_blender_path(blender_path)
            if resolved:
                self.blender_path = resolved

            if not self.blender_path or not os.path.exists(self.blender_path):
                error_msg = f"Error: Blender not found: '{self.blender_path}'. Set BLENDER_PATH or install Blender."
                log.warning("%s", error_msg)
                return (error_msg,)

            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)

            if not filename.lower().endswith('.glb'):
                filename = filename + '.glb'

            output_file = output_path / filename

            with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp_input:
                temp_input_path = tmp_input.name

            trimesh.export(temp_input_path, file_type='glb')
            log.info("Saved temporary input mesh to: %s", temp_input_path)

            blender_script = f"""
import bpy
import sys

try:
    print("Starting UV unwrap process...")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    print("Cleared scene")

    print("Importing GLB file...")
    bpy.ops.import_scene.gltf(filepath=r'{temp_input_path}')
    print(f"Imported {{len(bpy.context.selected_objects)}} objects")

    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if not mesh_objects:
        print("ERROR: No mesh objects found in the imported file!")
        sys.exit(1)

    print(f"Found {{len(mesh_objects)}} mesh object(s)")

    for obj in mesh_objects:
        print(f"Processing mesh: {{obj.name}}")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        print(f"Applying Smart UV Project (angle_limit={{angle_limit}}, island_margin={{island_margin}})...")
        bpy.ops.uv.smart_project(
            angle_limit={angle_limit},
            island_margin={island_margin},
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=False
        )
        print("UV unwrapping completed")

        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')

    print("Exporting GLB file...")
    bpy.ops.export_scene.gltf(
        filepath=r'{str(output_file)}',
        export_format='GLB'
    )

    print("Export completed successfully!")
    print("Output saved to: " + r'{str(output_file)}')

except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_script:
                tmp_script.write(blender_script)
                script_path = tmp_script.name

            log.info("Starting Blender UV unwrap process... Input: %s Output: %s",
                     temp_input_path, output_file)

            result = subprocess.run(
                [self.blender_path, "--background", "--python", script_path],
                capture_output=True,
                text=True,
                check=False
            )

            try:
                os.remove(temp_input_path)
                os.remove(script_path)
            except OSError:
                pass

            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                success_msg = f"UV unwrapped mesh saved to: {output_file} ({file_size:,} bytes)"
                log.info("%s", success_msg)
                log.info("--- Blender Output ---\n%s", result.stdout)
                return (str(output_file),)
            else:
                error_msg = f"Error: Output file was not created.\n\nBlender output:\n{result.stdout}\n\nBlender errors:\n{result.stderr}"
                log.warning("%s", error_msg)
                return (error_msg,)

        except (AttributeError, TypeError):
            error_msg = "Error: Input is not a valid trimesh object. Expected TRIMESH output."
            log.warning("%s", error_msg)
            return (error_msg,)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
            error_msg = f"Error during UV unwrapping: {str(e)}"
            log.exception("Error during UV unwrapping")
            return (error_msg,)
        except Exception as e:
            error_msg = f"Error during UV unwrapping: {str(e)}"
            log.exception("Error during UV unwrapping")
            return (error_msg,)


NODE_CLASS_MAPPINGS = {
    "JarJarSaveGLB": JarJarSaveGLB,
    "JarJarUVUnwrap": JarJarUVUnwrap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarSaveGLB": "JarJar Save GLB",
    "JarJarUVUnwrap": "JarJar UV Unwrap",
}
