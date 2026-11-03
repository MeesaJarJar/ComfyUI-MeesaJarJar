# MeesaJarJar Demo Workflows
Open any JSON in ComfyUI: drag the file onto the canvas, or
use the menu: `Load` button -> `Load` -> pick the workflow file.
1. `01_image_sprite_fx.json` - folder -> Pixelate -> PaletteReduce -> Outline -> Shadow -> save/preview.
2. `02_auto_segment_sprites.json` - folder -> AutoSegment (threshold/min_area/padding) -> Preview + Save sprites, Preview masks.
3. `03_splat_to_mesh.json` - splat PLY -> textured mesh (GLB) + 3D preview + GLB save.
4. `04_mesh_orient_preview.json` - load mesh -> Orient (rotate_x/y/z) -> 3D preview + image preview + framed camera.
5. `05_audio_text_utils.json` - audio load/save chain, text list -> save text, range -> list item.
All `EXAMPLE` / `D:\3DCATCHALL\example.ply` paths are placeholders: edit the
path/folder widgets to point at real files before queueing.
Workflows 03/04 need a real `.ply` / mesh file and (for 03) pymeshlab + xatlas installed.
Nodes used are all registered in `nodes/__init__.py` (`NODE_CLASS_MAPPINGS`)
plus core `LoadImage`, `PreviewImage`, `SaveImage`.
In 03, LoadSplatPly and SplatToMesh share the same file path on purpose:
one gives a quick Poisson preview, the other bakes the textured export file.
Mask preview in 02 wires a MASK output to PreviewImage; swap in a mask viewer if preferred.
Queue with `Ctrl+Enter`; missing-file errors name the widget path to fix.
Tip: duplicate the PreviewMesh node to compare before/after orientation side by side.
Tip: JarJarSaveImage writes `;`-separated explicit paths, core SaveImage uses a prefix.
Format reference: ComfyUI 0.24 UI graph (`version` 0.4, `nodes`/`links`/`groups`/`extra`).
