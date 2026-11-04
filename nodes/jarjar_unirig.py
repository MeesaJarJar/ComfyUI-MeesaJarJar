"""JarJar UniRig-compatible auto-rig node (host-safe).

Uses UniRig skeleton definitions (Mixamo/Vroid bone names pulled from
ComfyUI-UniRig) but runs WITHOUT the ML stack: no spconv, torch-scatter,
flash-attn, or comfy-env isolation. Everything runs in the host Portable
env (torch 2.13 / py3.13) plus Blender background mode for skinning.

Pipeline: TRIMESH -> temp GLB -> Blender background (import, build
proportion-fitted armature, parent with automatic weights) -> rigged FBX.
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


# UniRig bone names, pulled from
# ComfyUI-UniRig/nodes/unirig/configs/skeleton/mixamo.yaml + vroid.yaml
MIXAMO_BODY = [
    "mixamorig:Hips",
    "mixamorig:Spine",
    "mixamorig:Spine1",
    "mixamorig:Spine2",
    "mixamorig:Neck",
    "mixamorig:Head",
    "mixamorig:LeftShoulder",
    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",
    "mixamorig:LeftHand",
    "mixamorig:RightShoulder",
    "mixamorig:RightArm",
    "mixamorig:RightForeArm",
    "mixamorig:RightHand",
    "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg",
    "mixamorig:LeftFoot",
    "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg",
    "mixamorig:RightLeg",
    "mixamorig:RightFoot",
    "mixamorig:RightToeBase",
]

VROID_BODY = [
    "J_Bip_C_Hips",
    "J_Bip_C_Spine",
    "J_Bip_C_Chest",
    "J_Bip_C_UpperChest",
    "J_Bip_C_Neck",
    "J_Bip_C_Head",
    "J_Bip_L_Shoulder",
    "J_Bip_L_UpperArm",
    "J_Bip_L_LowerArm",
    "J_Bip_L_Hand",
    "J_Bip_R_Shoulder",
    "J_Bip_R_UpperArm",
    "J_Bip_R_LowerArm",
    "J_Bip_R_Hand",
    "J_Bip_L_UpperLeg",
    "J_Bip_L_LowerLeg",
    "J_Bip_L_Foot",
    "J_Bip_L_ToeBase",
    "J_Bip_R_UpperLeg",
    "J_Bip_R_LowerLeg",
    "J_Bip_R_Foot",
    "J_Bip_R_ToeBase",
]


class JarJarUnirig:
    """Auto-rig TRIMESH to a UniRig-named armature via Blender (no ML deps)."""

    def __init__(self):
        # Blender 4.4+ tested; resolved at runtime via _resolve_blender_path().
        self.blender_path = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {}),
                "output_folder": ("STRING", {"default": "./output", "multiline": False}),
                "filename": ("STRING", {"default": "rigged_model.fbx", "multiline": False}),
            },
            "optional": {
                "blender_path": ("STRING", {"default": "", "multiline": False}),
                "skeleton_template": (["mixamo", "vroid"], {"default": "mixamo"}),
                "auto_weights": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_output_path",)
    FUNCTION = "auto_rig"
    CATEGORY = "JarJar/3D"
    DESCRIPTION = (
        "UniRig-compatible auto-rig (host-safe): builds a Mixamo/Vroid-named "
        "armature proportion-fitted to the mesh and skins with Blender "
        "automatic weights. No spconv/flash-attn/comfy-env needed."
    )

    def auto_rig(self, trimesh, output_folder: str, filename: str,
                 blender_path: str = None, skeleton_template: str = "mixamo",
                 auto_weights: bool = True):
        try:
            resolved = _resolve_blender_path(blender_path)
            if resolved:
                self.blender_path = resolved

            if not self.blender_path or not os.path.exists(self.blender_path):
                msg = f"Error: Blender not found: '{self.blender_path}'. Set BLENDER_PATH or install Blender."
                log.warning("%s", msg)
                return (msg,)

            bones = MIXAMO_BODY if skeleton_template != "vroid" else VROID_BODY
            # parent map over the 22 body bones (index-based)
            parents = {
                1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
                6: 3, 7: 6, 8: 7, 9: 8,
                10: 3, 11: 10, 12: 11, 13: 12,
                14: 0, 15: 14, 16: 15, 17: 16,
                18: 0, 19: 18, 20: 19, 21: 20,
            }
            parent_map_str = "{" + ", ".join(f"{k}: {v}" for k, v in parents.items()) + "}"
            bone_list_str = "[" + ", ".join(f'"{b}"' for b in bones) + "]"

            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / filename
            if output_file.suffix.lower() != ".fbx":
                output_file = output_file.with_suffix(".fbx")

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_in:
                temp_input_path = tmp_in.name
            trimesh.export(temp_input_path, file_type="glb")

            op = "ARMATURE" if auto_weights else "ENVELOPE"

            blender_script = f"""import bpy, sys
from mathutils import Vector
try:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    print("Cleared scene")
    bpy.ops.import_scene.gltf(filepath=r'{temp_input_path}')
    print(f"Imported {{len(bpy.context.selected_objects)}} objects")
    mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objects:
        print("ERROR: No mesh objects found!")
        sys.exit(1)
    print(f"Found {{len(mesh_objects)}} mesh object(s)")
    # bounds over all meshes (world space)
    mins = [1e9, 1e9, 1e9]; maxs = [-1e9, -1e9, -1e9]
    for o in mesh_objects:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mins[i] = min(mins[i], w[i]); maxs[i] = max(maxs[i], w[i])
    cx = (mins[0] + maxs[0]) / 2.0
    cy = (mins[1] + maxs[1]) / 2.0
    minz, maxz = mins[2], maxs[2]
    h = max(maxz - minz, 1e-6)
    half_w = max((maxs[0] - mins[0]) / 2.0, h * 0.05)
    print(f"Bounds h={{h:.3f}} cx={{cx:.3f}} cy={{cy:.3f}} z=[{{minz:.3f}}, {{maxz:.3f}}]")
    BONES = {bone_list_str}
    PARENTS = {parent_map_str}
    # proportional joint layout (fractions of height, x as fraction of half-width)
    L = {{
        0: (0.0, 0.0, 0.55), 1: (0.0, 0.0, 0.62), 2: (0.0, 0.0, 0.70),
        3: (0.0, 0.0, 0.78), 4: (0.0, 0.0, 0.86), 5: (0.0, 0.0, 0.94),
        6: (0.35, 0.0, 0.78), 7: (0.75, 0.0, 0.78), 8: (1.25, 0.0, 0.78),
        9: (1.65, 0.0, 0.78),
        10: (-0.35, 0.0, 0.78), 11: (-0.75, 0.0, 0.78), 12: (-1.25, 0.0, 0.78),
        13: (-1.65, 0.0, 0.78),
        14: (0.28, 0.0, 0.52), 15: (0.30, 0.0, 0.28), 16: (0.30, 0.05, 0.04),
        17: (0.30, 0.18, 0.0),
        18: (-0.28, 0.0, 0.52), 19: (-0.30, 0.0, 0.28), 20: (-0.30, 0.05, 0.04),
        21: (-0.30, 0.18, 0.0),
    }}
    bpy.ops.object.armature_add(enter_editmode=True, location=(cx, cy, minz))
    arm = bpy.context.active_object
    arm.name = "JarJarUnirig_Armature"
    eb = arm.data.edit_bones
    for b in list(eb):
        eb.remove(b)
    created = {{}}
    for i, name in enumerate(BONES):
        fx, fy, fz = L[i]
        head = (cx + fx * half_w, cy + fy * half_w, minz + fz * h)
        # tail: toward first child if any, else small offset
        kids = [k for k, v in PARENTS.items() if v == i]
        if kids:
            kfx, kfy, kfz = L[kids[0]]
            tail = (cx + kfx * half_w, cy + kfy * half_w, minz + kfz * h)
        else:
            tail = (head[0], head[1] + half_w * 0.1, head[2] + h * 0.02)
        b = eb.new(name)
        b.head = head; b.tail = tail
        created[i] = b
    for k, v in PARENTS.items():
        created[k].parent = created[v]
        created[k].use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Armature built: {{len(created)}} bones")
    # skin
    bpy.ops.object.select_all(action='DESELECT')
    for o in mesh_objects:
        o.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='{op}')
    print(f"Parented with {{'{op}'}} weights")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.fbx(
        filepath=r'{str(output_file)}',
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=False,
    )
    print("FBX export done: " + r'{str(output_file)}')
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback; traceback.print_exc()
    sys.exit(1)
"""

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_script:
                tmp_script.write(blender_script)
                script_path = tmp_script.name

            log.info("JarJarUnirig: rigging with %s template (%d bones)...", skeleton_template, len(bones))
            result = subprocess.run(
                [self.blender_path, "--background", "--python", script_path],
                capture_output=True, text=True, check=False,
            )

            for f in (temp_input_path, script_path):
                try:
                    os.remove(f)
                except OSError:
                    pass

            if os.path.exists(output_file):
                msg = f"Rigged FBX saved to: {output_file} ({os.path.getsize(output_file):,} bytes)"
                log.info("%s", msg)
                log.info("--- Blender Output ---\n%s", result.stdout[-3000:])
                return (str(output_file),)
            msg = f"Error: FBX not created.\nBlender out:\n{result.stdout[-2000:]}\nBlender err:\n{result.stderr[-2000:]}"
            log.warning("%s", msg)
            return (msg,)

        except (AttributeError, TypeError):
            msg = "Error: Input is not a valid trimesh object. Expected TRIMESH output."
            log.warning("%s", msg)
            return (msg,)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
            log.exception("Error during UniRig-compatible rigging")
            msg = f"Error during UniRig-compatible rigging: {e}"
            return (msg,)
        except Exception as e:
            log.exception("Error during UniRig-compatible rigging")
            msg = f"Error during UniRig-compatible rigging: {e}"
            return (msg,)


NODE_CLASS_MAPPINGS = {
    "JarJarUnirig": JarJarUnirig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarUnirig": "JarJar UniRig Auto-Rig",
}
