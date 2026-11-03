# ComfyUI-MeesaJarJar

Unified pack merging all JarJar node sets. Base path: `ComfyUI/custom_nodes/ComfyUI-MeesaJarJar/nodes/`.

## Audio

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| JarJarAudioSaveAudio | JarJar Audio Save Audio | `nodes/audio_nodes.py` | Save audio tensor to disk. |
| JarJarAudioLoadSample | JarJar Audio Load Sample | `nodes/audio_nodes.py` | Load audio file as tensor. |

## IO

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| LoadImageByIndex | Load Image By Index | `nodes/image_io_nodes.py` | Load single image from folder by index. |
| JarJarLoadImageFolder | JarJar Load Image Folder | `nodes/image_io_nodes.py` | Load images from a folder by index. |
| JarJarMeesaLoadImagesFromFolder | JarJar Load Images From Folder (Meesa) | `nodes/image_io_nodes.py` | Rich folder loader (owns no canonical key). |
| JarJarSaveImage | JarJar Save Image | `nodes/image_io_nodes.py` | Save IMAGE with options. |
| JarJarLoadImagesFromFolder | JarJar: Load Images From Folder | `nodes/classic_nodes.py` (`classic_load_images_from_folder.py`) | Classic simple folder loader; owns canonical key. |

## Image

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| JarJarResizeImage | JarJar Resize Image | `nodes/image_ops_nodes.py` | Resize IMAGE tensor. |
| JarJarTileImage | JarJar Tile Image | `nodes/image_ops_nodes.py` | Tile images into a grid. |
| JarJarPixelizeImage | JarJar Pixelize Image | `nodes/image_ops_nodes.py` | Pixel-sprite effect (needs optional `pixeloe`). |
| JarJarMakeShadow | JarJar Make Shadow (Auto-Crop) | `nodes/image_ops_nodes.py` | Generate shadow from image. |
| JarJarCompareImages | JarJar: Compare Images (PSNR/SSIM) | `nodes/image_metrics_nodes.py` | Score two images with PSNR/SSIM. |
| JarJarIntToString | JarJar: INT to STRING TEXT | `nodes/classic_nodes.py` (`classic_int_to_string.py`) | Format int as string. |
| JarJarPixelate | JarJar: Pixelate | `nodes/classic_nodes.py` (`classic_pixelate.py`) | Downscale/upscale pixelation. |
| JarJarPaletteReduce | JarJar: Palette Reduce | `nodes/classic_nodes.py` (`classic_palette_reduce.py`) | Reduce image to N colors. |
| JarJarDither | JarJar: Dither | `nodes/classic_nodes.py` (`classic_dither.py`) | Ordered dithering. |
| JarJarOutline | JarJar: Outline | `nodes/classic_nodes.py` (`classic_outline.py`) | Edge outline overlay. |
| JarJarBrightnessContrast | JarJar: Brightness/Contrast | `nodes/classic_nodes.py` (`classic_brightness_contrast.py`) | Adjust brightness/contrast. |
| JarJarRemoveBackground | JarJar: Remove Background | `nodes/classic_nodes.py` (`classic_remove_background.py`) | Dark pixels to transparent; outputs RGBA. |
| JarJarShadow | JarJar: Shadow | `nodes/classic_nodes.py` (`classic_shadow.py`) | Drop shadow effect. |

## Utils

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| ListFilesNode | List Files Node | `nodes/list_utils_nodes.py` | List files in a folder. |
| JarJarFolderFilesByExt | JarJar: Folder Files by Extension | `nodes/list_utils_nodes.py` | List folder files filtered by extension. |
| JarJarTextList | JarJar Text List | `nodes/list_utils_nodes.py` | Build a text list. |
| JarJarCSVFileToList | JarJar CSV File To List | `nodes/list_utils_nodes.py` | Read CSV file into list. |
| JarJarListItem | JarJar List Item | `nodes/list_utils_nodes.py` | Pick item from list by index. |
| JarJarRangeIntLarge | JarJar Range Int Large | `nodes/list_utils_nodes.py` | Large-range int generator. |

## Text

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| JarJarSaveText | JarJar Save Text | `nodes/text_nodes.py` | Save string to text file. |

## 3D-Mesh

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| JarJarSaveGLB | JarJar Save GLB | `nodes/mesh_nodes.py` | Export TRIMESH to .glb file. |
| JarJarUVUnwrap | JarJar UV Unwrap | `nodes/mesh_nodes.py` | UV unwrap via Blender. |
| JarJarLoadMesh | JarJar: Load Mesh | `nodes/classic_nodes.py` (`classic_load_mesh.py`) | Load mesh/splat/SPZ file as TRIMESH, splats as vertex-colored points. |
| JarJarPreviewMesh | JarJar: Preview Mesh | `nodes/mesh_preview_nodes.py` | View TRIMESH/file in 3D viewport, passthrough. |
| JarJarFrameCamera | JarJar: Frame Camera on Mesh | `nodes/mesh_preview_nodes.py` | Build camera framing a TRIMESH. |
| JarJarPreviewMeshImage | JarJar: Preview Mesh (Image) | `nodes/mesh_preview_nodes.py` | Render shaded orthographic IMAGE views. |
| JarJarOrientMesh | JarJar: Orient Mesh | `nodes/orient_mesh.py` | Rotate mesh about X/Y/Z degrees. |
| JarJarSplatToMesh | JarJar: Splat PLY to Textured Mesh | `nodes/splat_mesh_export_nodes.py` | Reconstruct textured mesh from splat PLY. |
| JarJarSplatToFilePath | JarJar: Splat to File + Path | `nodes/splat_path_nodes.py` | Write SPLAT to file, return path. |
| JarJarSubmit3D | Submit 3D to JarJar Foundry | `nodes/foundry_nodes.py` (`foundry_submit.py`) | Submit 3D file to Foundry service. |
| MeesaAssetsPushImage | Push Image to MeesaAssets | `nodes/meesa_assets_nodes.py` | Upload IMAGE to MeesaAssets API. |
| MeesaAssetsPushModel | Push 3D Model to MeesaAssets | `nodes/meesa_assets_nodes.py` | Upload model file to MeesaAssets API. |
| MeesaAssetsPullImage | Pull Image from MeesaAssets | `nodes/meesa_assets_nodes.py` | Download image asset as IMAGE. |
| MeesaAssetsPushModel3D | Push 3D (File3D) to MeesaAssets | `nodes/meesa_assets_nodes.py` | Upload File3D link to MeesaAssets API. |

## 3D-Splat

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| JarJarLoadSplatPly | JarJar: Load Splat PLY as Mesh | `nodes/splat_ply_nodes.py` | Load gaussian-splat PLY as textured TRIMESH. |
| JarJarPreviewSplatPly | JarJar: Preview Splat PLY | `nodes/splat_ply_preview_nodes.py` | Preview splat PLY in 3D viewport. |
| JarJarSplatAnchorTexture | JarJar: Splat Anchor Texture (front) | `nodes/splat_retexture_nodes.py` | Retexture splats from one front image. |
| JarJarSplatMultiviewTexture | JarJar: Splat Multiview Texture (orbit) | `nodes/splat_retexture_nodes.py` | Retexture splats from orbit views. |
| JarJarSplatBakeVertexTexture | JarJar: Splat Bake Mesh Texture | `nodes/splat_retexture_nodes.py` | Copy mesh vertex colors onto splats. |
| JarJarSplatFillLight | JarJar: Splat Fill Light | `nodes/splat_fill_nodes.py` | Lift dark splats via SH DC term. |
| JarJarUpscaleSplatsSubdivide | JarJar: Upscale Splats (Subdivide) | `nodes/splat_upscale_nodes.py` | Densify splats by subdivision. |
| JarJarUpscaleSplatsAdaptive | JarJar: Upscale Splats (Adaptive Detail) | `nodes/splat_upscale_nodes.py` | Densify splats adaptively by detail. |
| JarJarDownsampleSplatsTopK | JarJar: Downsample Splats (Top-K) | `nodes/splat_upscale_nodes.py` | Keep top-K splats by opacity. |
| JarJarDownsampleSplatsVoxel | JarJar: Downsample Splats (Voxel Merge) | `nodes/splat_upscale_nodes.py` | Merge splats per voxel. |
| JarJarSplatReconstructScore | JarJar: Splat Reconstruction Score | `nodes/splat_upscale_nodes.py` | Score splat render quality. |
| JarJarSplatEnhance | JarJar: Splat Enhance | `nodes/splat_upscale_nodes.py` | Enhance splat sharpness/detail. |

## Rig

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| JarJarUnirig | JarJar UniRig Auto-Rig | `nodes/jarjar_unirig.py` | Auto-rig TRIMESH with UniRig. |

## Segment

| Key | Display | File | Purpose |
| --- | --- | --- | --- |
| JarJarAutoSegmentImage | JarJar Auto Segment Image | `nodes/jarjar_auto_segment_image.py` | Split image into sprite crops + masks. |

## Retired packs

Retired `ComfyUI-JarJarNodes`: all 10 keys now live in `nodes/classic_*.py` via `nodes/classic_nodes.py` (`JarJarIntToString`, `JarJarLoadMesh`, `JarJarPixelate`, `JarJarPaletteReduce`, `JarJarDither`, `JarJarOutline`, `JarJarBrightnessContrast`, `JarJarLoadImagesFromFolder`, `JarJarRemoveBackground`, `JarJarShadow`). `Orient` ported as `JarJarOrientMesh` (`nodes/orient_mesh.py`). `PreviewMesh` split into viewport (`JarJarPreviewMesh`) plus image (`JarJarPreviewMeshImage`, both in `nodes/mesh_preview_nodes.py`).

## Notes

* `JarJarLoadMesh` upgrade: loads meshes plus SPZ and splat PLY as vertex-colored TRIMESH (open3d path when available).
* `JarJarRemoveBackground` fix: outputs RGBA (alpha from threshold mask) instead of dropping alpha.

## Installation

Via ComfyUI Manager: search for "MeesaJarJar" and install. Or manual install:

```
cd ComfyUI/custom_nodes
git clone https://github.com/MeesaJarJar/ComfyUI-MeesaJarJar
```

Then install requirements (portable build: use the embedded python, e.g.
`ComfyUI_Portable/python_embeded/python.exe -m pip install -r requirements.txt`
run from `ComfyUI/custom_nodes/ComfyUI-MeesaJarJar/`):

```
pip install -r requirements.txt
```

Restart ComfyUI.

## License

Released into the public domain under the Unlicense — zero attribution required.
