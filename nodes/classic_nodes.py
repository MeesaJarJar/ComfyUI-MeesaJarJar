"""Classic JarJar nodes - merged verbatim from ComfyUI-JarJarNodes/nodes/*.py.

Source files copied as classic_*.py to avoid Windows case-collisions with Meesa files.
This wrapper re-exports them under their ORIGINAL node keys + display names.
"""

from .classic_brightness_contrast import JarJarBrightnessContrast
from .classic_dither import JarJarDither
from .classic_int_to_string import JarJarIntToString
from .classic_load_images_from_folder import JarJarLoadImagesFromFolder as JarJarClassicLoadImagesFromFolder
from .classic_load_mesh import JarJarLoadMesh
from .classic_outline import JarJarOutline
from .classic_palette_reduce import JarJarPaletteReduce
from .classic_pixelate import JarJarPixelate
from .classic_remove_background import JarJarRemoveBackground
from .classic_shadow import JarJarShadow

NODE_CLASS_MAPPINGS = {
    "JarJarIntToString": JarJarIntToString,
    "JarJarLoadMesh": JarJarLoadMesh,
    "JarJarPixelate": JarJarPixelate,
    "JarJarPaletteReduce": JarJarPaletteReduce,
    "JarJarDither": JarJarDither,
    "JarJarOutline": JarJarOutline,
    "JarJarBrightnessContrast": JarJarBrightnessContrast,
    "JarJarLoadImagesFromFolder": JarJarClassicLoadImagesFromFolder,
    "JarJarRemoveBackground": JarJarRemoveBackground,
    "JarJarShadow": JarJarShadow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarIntToString": "JarJar: INT to STRING TEXT",
    "JarJarLoadMesh": "JarJar: Load Mesh",
    "JarJarPixelate": "JarJar: Pixelate",
    "JarJarPaletteReduce": "JarJar: Palette Reduce",
    "JarJarDither": "JarJar: Dither",
    "JarJarOutline": "JarJar: Outline",
    "JarJarBrightnessContrast": "JarJar: Brightness/Contrast",
    "JarJarLoadImagesFromFolder": "JarJar: Load Images From Folder",
    "JarJarRemoveBackground": "JarJar: Remove Background",
    "JarJarShadow": "JarJar: Shadow",
}
