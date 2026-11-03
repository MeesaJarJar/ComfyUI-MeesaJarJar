import os
import torch
import numpy as np
from PIL import Image

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


class JarJarLoadImagesFromFolder:
    """Load all images from a folder path as a batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filenames")
    FUNCTION = "load"
    CATEGORY = "JarJar/IO"
    OUTPUT_NODE = False

    def load(self, path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Folder not found: {path}")

        files = sorted([
            f for f in os.listdir(path)
            if f.lower().endswith(VALID_EXT)
        ])

        if not files:
            raise FileNotFoundError(f"No images found in: {path}")

        images = []
        names = []
        for f in files:
            img = Image.open(os.path.join(path, f)).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(torch.from_numpy(arr))
            names.append(f)

        batch = torch.stack(images, dim=0)
        filename_str = "\n".join(names)
        return (batch, filename_str)
