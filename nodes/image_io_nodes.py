"""MeesaJarJar image IO nodes - extracted from Old pack:
- jarjar_load_image_path.py (LoadImageByIndex)
- JarJarLoadImageFolder.py
- JarJarLoadImagesFromFolder.py (renamed key to avoid collision with ComfyUI-JarJarNodes)
- jarjar_save_image.py (JarJarSaveImage)
"""

import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps

log = logging.getLogger("jarjar")


class JarJarLoadImageByIndex:
    """Load a single image from a directory by sorted index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "load"
    CATEGORY = "JarJar/IO"
    DESCRIPTION = "Loads the Nth sorted image (.png/.jpg/.jpeg/.bmp/.webp) from directory. Returns IMAGE + filename."

    def load(self, directory, index):
        if not os.path.isdir(directory):
            raise RuntimeError(f"Directory does not exist: {directory}")

        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        files.sort()
        if not files:
            raise RuntimeError("No image files found in directory.")

        if index < 0 or index >= len(files):
            raise RuntimeError(f"Index {index} is out of range (directory has {len(files)} images).")

        filename = files[index]
        filepath = os.path.join(directory, filename)
        try:
            img = Image.open(filepath).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Could not load image at {filepath}: {e}")

        return (tensor, filename)


class JarJarLoadImageFolder:
    """Load images from a directory with cap / start_index / auto-increment."""

    def __init__(self):
        self.auto_increment_counter = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "image_load_cap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                }),
                "load_always": (["enabled", "disabled"],),
                "auto_increment": (["enabled", "disabled"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "FILE_PATH")
    FUNCTION = "load_images"
    CATEGORY = "JarJar/IO"
    DESCRIPTION = "Loads sorted images from directory with cap + start_index + optional auto-increment. Returns IMAGE batch + ;-joined paths."

    def load_images(self, directory, image_load_cap, start_index, load_always, auto_increment):
        try:
            if auto_increment == "enabled":
                start_index = start_index + self.auto_increment_counter
                self.auto_increment_counter += 1
            if not os.path.isdir(directory):
                raise ValueError(f"Directory does not exist: {directory}")

            supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff'}
            image_files = []

            for filename in sorted(os.listdir(directory)):
                if os.path.splitext(filename)[1].lower() in supported_formats:
                    image_files.append(os.path.join(directory, filename))

            if not image_files:
                raise ValueError(f"No images found in directory: {directory}")

            image_files = image_files[start_index:]

            if image_load_cap > 0:
                image_files = image_files[:image_load_cap]

            images = []
            file_paths = []

            for image_path in image_files:
                try:
                    img = Image.open(image_path).convert('RGB')
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(torch.from_numpy(img_array))
                    file_paths.append(image_path)
                except (OSError, ValueError) as e:
                    log.info("Error loading image %s: %s", image_path, e)
                    continue

            if not images:
                raise ValueError("Failed to load any images from directory")

            image_tensor = torch.stack(images, dim=0)
            file_path_str = ";".join(file_paths)

            log.info("Loaded %d images from %s", len(images), directory)
            return (image_tensor, file_path_str)

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Error loading images: {str(e)}"
            log.info("%s", error_msg)
            raise RuntimeError(error_msg)


class JarJarMeesaLoadImagesFromFolder:
    """Load all images from folder with cap + start_index. Returns batch + masks + count + paths.

    NOTE: Old pack registered this as "JarJarLoadImagesFromFolder", which collides with
    ComfyUI-JarJarNodes' node of the same name. This suite registers the unique key
    "JarJarMeesaLoadImagesFromFolder" (class kept compatible) to avoid shadowing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "count", "image_path", "image_filename")
    FUNCTION = "load_images"
    CATEGORY = "JarJar/IO"
    DESCRIPTION = "Loads .jpg/.jpeg/.png/.webp from folder (EXIF-corrected, alpha->mask, mismatched sizes auto-resized to first)."

    def load_images(self, folder, image_load_cap=0, start_index=0):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")

        dir_files = os.listdir(folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(folder, x) for x in dir_files]

        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_path_list = []
        image_filename_list = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break

            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]

                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                    has_non_empty_mask = True
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

                images.append(image)
                masks.append(mask)
                image_path_list.append(image_path)
                image_filename_list.append(os.path.basename(image_path))
                image_count += 1
            except (OSError, ValueError) as e:
                log.info("Error loading %s: %s", image_path, e)
                continue

        if len(images) == 0:
            raise RuntimeError(f"No valid images loaded from '{folder}'")

        if len(images) == 1:
            return (images[0], masks[0], 1, image_path_list[0], image_filename_list[0])

        elif len(images) > 1:
            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2_resized = torch.nn.functional.interpolate(
                        image2.movedim(-1, 1),
                        size=(image1.shape[1], image1.shape[2]),
                        mode='bilinear',
                        align_corners=False
                    ).movedim(1, -1)
                    image1 = torch.cat((image1, image2_resized), dim=0)
                else:
                    image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks[1:]:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(
                            mask2.unsqueeze(0).unsqueeze(0),
                            size=(image1.shape[1], image1.shape[2]),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

            filenames_str = ";".join(image_filename_list)
            image_paths_str = ";".join(image_path_list)

            return (image1, mask1, len(images), image_paths_str, filenames_str)


class JarJarSaveImage:
    """Save IMAGE batch to ;-separated explicit file paths."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filenames": ("STRING", {"default": "output/test.png"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "save_image"
    CATEGORY = "JarJar/IO"
    DESCRIPTION = "Saves each batch image to the ;-separated filenames list (auto-prefixes output/, auto-adds .png)."
    OUTPUT_NODE = True

    def save_image(self, image, filenames):
        if isinstance(image, list):
            image = image[0]

        if isinstance(image, torch.Tensor):
            img_batch = image.detach().cpu().numpy()
        else:
            img_batch = np.array(image)

        if img_batch.ndim == 4:
            batch_size = img_batch.shape[0]
        elif img_batch.ndim == 3:
            img_batch = np.expand_dims(img_batch, 0)
            batch_size = 1
        else:
            return (f"Error: Invalid image dimensions {img_batch.ndim}",)

        filename_list = [f.strip() for f in filenames.split(";")]

        saved_paths = []

        for idx in range(batch_size):
            img_np = img_batch[idx]

            if idx < len(filename_list):
                current_filename = filename_list[idx]
            else:
                continue

            if not current_filename.lower().endswith(".png"):
                current_filename += ".png"

            if not current_filename.startswith("output/"):
                current_filename = "output/" + current_filename

            folder = os.path.dirname(current_filename)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4) and img_np.shape[2] not in (1, 3, 4):
                img_np = np.transpose(img_np, (1, 2, 0))

            if np.issubdtype(img_np.dtype, np.floating):
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            elif img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)

            try:
                im = Image.fromarray(img_np)
            except (OSError, ValueError, TypeError):
                continue

            try:
                im.save(current_filename)
                saved_paths.append(current_filename)
            except (OSError, ValueError):
                continue

        result_path = ", ".join(saved_paths) if saved_paths else "No images saved"
        return (result_path,)


NODE_CLASS_MAPPINGS = {
    "JarJarLoadImageByIndex": JarJarLoadImageByIndex,
    "JarJarLoadImageFolder": JarJarLoadImageFolder,
    "JarJarMeesaLoadImagesFromFolder": JarJarMeesaLoadImagesFromFolder,
    "JarJarSaveImage": JarJarSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarLoadImageByIndex": "JarJar Load Image By Index",
    "JarJarLoadImageFolder": "JarJar Load Image Folder",
    "JarJarMeesaLoadImagesFromFolder": "JarJar Load Images From Folder (Meesa)",
    "JarJarSaveImage": "JarJar Save Image",
}
