"""MeesaJarJar image-ops nodes - extracted from Old pack:
- JarJarResizeImage.py
- JarJarTileImage (extension-less file in Old pack, fixed here)
- JarJarPixelize.py (optional pixeloe dep)
- JarJarShadow.py (JarJarMakeShadow)
"""

import math

import numpy as np
import torch
from PIL import Image, ImageFilter

try:
    from pixeloe.torch.pixelize import pixelize as px_pixelize
except ImportError:
    px_pixelize = None


class JarJarResizeImage:
    """Center-crop / center-paste image onto exact WxH transparent canvas. Returns IMAGE + MASK."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("STRING", {"default": "512", "multiline": False}),
                "height": ("STRING", {"default": "512", "multiline": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "resize"
    CATEGORY = "JarJar/Image"
    DESCRIPTION = "Center-crops if larger, center-pastes on transparent WxH canvas. Width/height are STRING so they can be wired."

    def resize(self, image, width, height, mask=None):
        w = int(str(width).strip())
        h = int(str(height).strip())

        img = image.detach().cpu().numpy()

        if img.ndim == 3:
            img = img[None, ...]

        out_images = []
        out_masks = []

        for im in img:
            if im.shape[0] in (1, 3, 4) and im.shape[-1] not in (1, 3, 4):
                im = np.transpose(im, (1, 2, 0))

            if im.shape[-1] == 3:
                alpha = np.ones((im.shape[0], im.shape[1], 1), dtype=im.dtype)
                im = np.concatenate([im, alpha], axis=-1)

            rgb = im[..., :3]
            alpha = im[..., 3:4]

            alpha_safe = np.clip(alpha, 1e-6, 1.0)
            rgb = rgb / alpha_safe
            rgb = np.clip(rgb, 0.0, 1.0)

            rgba = np.concatenate([rgb, alpha], axis=-1)

            im_uint8 = (rgba * 255).astype(np.uint8)
            pil = Image.fromarray(im_uint8, "RGBA")

            iw, ih = pil.size

            if iw > w or ih > h:
                left = max(0, (iw - w) // 2)
                top = max(0, (ih - h) // 2)
                right = left + min(iw, w)
                bottom = top + min(ih, h)
                pil = pil.crop((left, top, right, bottom))
                iw, ih = pil.size

            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            x = (w - iw) // 2
            y = (h - ih) // 2
            canvas.paste(pil, (x, y), pil)

            out = np.array(canvas).astype(np.float32) / 255.0
            out_images.append(out)
            out_masks.append(out[..., 3])

        return (
            torch.from_numpy(np.stack(out_images, axis=0)),
            torch.from_numpy(np.stack(out_masks, axis=0)),
        )


class JarJarTileImage:
    """Tile one image into rows x columns grid on WxH canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "tile"
    CATEGORY = "JarJar/Image"
    DESCRIPTION = "Resizes source to tile size (W/columns x H/rows, LANCZOS) and repeats it rows x columns. RGBA composited on white."

    def tile(self, image, rows, columns, width, height):
        img = image.detach().cpu().numpy()
        if img.ndim == 3:
            img = img[None, ...]

        out_images = []

        for im in img:
            if im.shape[0] in (1, 3, 4) and im.shape[-1] not in (1, 3, 4):
                im = np.transpose(im, (1, 2, 0))

            if im.shape[-1] == 1:
                im = np.repeat(im, 3, axis=-1)
            elif im.shape[-1] == 4:
                rgb = im[..., :3]
                alpha = im[..., 3:4]
                rgb = rgb * alpha + (1 - alpha)
                im = np.clip(rgb, 0.0, 1.0)

            im_uint8 = (im * 255).astype(np.uint8)
            pil = Image.fromarray(im_uint8, "RGB")

            tile_w = width // columns
            tile_h = height // rows

            pil_resized = pil.resize((tile_w, tile_h), Image.LANCZOS)

            canvas = Image.new("RGB", (width, height), (0, 0, 0))

            for row in range(rows):
                for col in range(columns):
                    x = col * tile_w
                    y = row * tile_h
                    canvas.paste(pil_resized, (x, y))

            out = np.array(canvas).astype(np.float32) / 255.0
            out_images.append(out)

        return (torch.from_numpy(np.stack(out_images, axis=0)),)


class JarJarPixelizeImage:
    """UO-style pixel sprite via pixeloe + palette/dither/outline/upscale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "target_size": ("INT", {"default": 32, "min": 8, "max": 2048, "step": 1}),
                "palette": ("STRING", {"default": "UO Default"}),
                "dither": ("BOOL", {"default": False}),
                "outline": ("BOOL", {"default": False}),
                "bg_transparent": ("BOOL", {"default": True}),
                "contrast": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "upscale": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "preview_steps": ("BOOL", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pixelized_image",)
    FUNCTION = "pixelize"
    CATEGORY = "JarJar/Image"
    DESCRIPTION = "Pixeloe pixelize to target_size + UO palette (216) + dither/outline/contrast/upscale. Needs 'pip install pixeloe'."

    def _convert_to_pil(self, img):
        if isinstance(img, Image.Image):
            return img
        elif isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = img.permute(1, 2, 0)
            if img.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.numpy())
        elif isinstance(img, np.ndarray):
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            return Image.fromarray(img.astype(np.uint8))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def _pil_to_tensor(self, pil_img):
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def _tensor_to_pil(self, tensor):
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.shape[0] in [1, 3, 4]:
            tensor = tensor.permute(1, 2, 0)
        img_array = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
        return Image.fromarray(img_array)

    def pixelize(
        self,
        image=None,
        target_size=32,
        palette="UO Default",
        dither=False,
        outline=False,
        bg_transparent=True,
        contrast=0,
        upscale=1,
        preview_steps=True
    ):
        if px_pixelize is None:
            raise RuntimeError("Missing optional dependency 'pixeloe'. Install with 'pip install pixeloe' to use the pixelize node.")

        if image is None:
            image = Image.new("RGB", (32, 32), (0, 0, 0))

        pil_image = self._convert_to_pil(image)

        max_side = max(pil_image.size)
        if max_side != target_size:
            scale_factor = target_size / max_side
            new_size = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))
            pil_image = pil_image.resize(new_size, Image.NEAREST)

        img_tensor = self._pil_to_tensor(pil_image)
        pixelized_tensor = px_pixelize(img_tensor)
        img_small = self._tensor_to_pil(pixelized_tensor)

        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img_small = img_small.point(lambda i: 128 + factor * (i - 128))

        if palette.lower() == "uo default":
            img_small = img_small.convert("P", palette=Image.ADAPTIVE, colors=216)

        if dither:
            img_small = img_small.convert("RGB").convert("1").convert("RGB")

        if outline:
            edges = img_small.filter(ImageFilter.FIND_EDGES)
            edge_mask = edges.convert("L").point(lambda x: 0 if x < 128 else 255)
            img_small = Image.composite(Image.new("RGB", img_small.size, (0, 0, 0)), img_small, edge_mask)

        if bg_transparent:
            img_small = img_small.convert("RGBA")
            datas = img_small.getdata()
            newData = [
                (0, 0, 0, 0) if item[:3] == (0, 0, 0) else item
                for item in datas
            ]
            img_small.putdata(newData)

        if upscale > 1:
            img_small = img_small.resize(
                (img_small.width * upscale, img_small.height * upscale), Image.NEAREST
            )

        if img_small.mode == 'RGBA':
            img_array = np.array(img_small).astype(np.float32) / 255.0
        else:
            img_array = np.array(img_small.convert('RGB')).astype(np.float32) / 255.0

        output_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return (output_tensor,)


class JarJarMakeShadow:
    """Isometric shadow from RGBA alpha with auto-crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle_deg": ("FLOAT", {"default": 30.0, "min": -80.0, "max": 80.0, "step": 1.0}),
                "vertical_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.5, "step": 0.05}),
                "offset_x": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "offset_y": ("INT", {"default": 50, "min": -512, "max": 512, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "shadow_opacity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "make_shadow"
    CATEGORY = "JarJar/Image"
    DESCRIPTION = "Builds sheared/scaled/blurred black shadow from alpha, composites under sprite, auto-crops transparency. First batch item only."

    def make_shadow(self, image, angle_deg, vertical_scale, offset_x, offset_y, blur_radius, shadow_opacity):
        img_t = image[0].detach().cpu()
        img_np = img_t.numpy()

        if img_np.ndim == 4:
            img_np = img_np[0]
        if img_np.shape[0] in [3, 4]:
            img_np = np.transpose(img_np, (1, 2, 0))

        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[2] == 3:
            alpha = np.full((img_np.shape[0], img_np.shape[1], 1), 255, dtype=np.uint8)
            img_np = np.concatenate([img_np, alpha], axis=2)

        pil_img = Image.fromarray(img_np, mode="RGBA")
        width, height = pil_img.size

        alpha = pil_img.getchannel("A")
        shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        black_mask = Image.new("RGBA", (width, height), (0, 0, 0, int(255 * shadow_opacity)))
        shadow.paste(black_mask, mask=alpha)

        angle_rad = math.radians(angle_deg)
        shear_x = -math.tan(angle_rad)
        new_height = int(height * vertical_scale)
        new_width = int(width + height * abs(shear_x))

        coeffs = (1, shear_x, 0, 0, vertical_scale, 0)
        shadow = shadow.transform(
            (new_width, new_height),
            Image.AFFINE,
            coeffs,
            resample=Image.BILINEAR,
        )

        if blur_radius > 0:
            shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))

        total_width = max(width, new_width + abs(offset_x))
        total_height = max(height, new_height + abs(offset_y))
        canvas = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))

        img_x = max(0, -offset_x)
        img_y = max(0, -offset_y)
        shadow_x = img_x + offset_x
        shadow_y = img_y + offset_y

        canvas.paste(shadow, (int(shadow_x), int(shadow_y)), shadow)
        canvas.paste(pil_img, (int(img_x), int(img_y)), pil_img)

        bbox = canvas.getbbox()
        if bbox:
            canvas = canvas.crop(bbox)

        result_np = np.array(canvas).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "JarJarResizeImage": JarJarResizeImage,
    "JarJarTileImage": JarJarTileImage,
    "JarJarPixelizeImage": JarJarPixelizeImage,
    "JarJarMakeShadow": JarJarMakeShadow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarResizeImage": "JarJar Resize Image",
    "JarJarTileImage": "JarJar Tile Image",
    "JarJarPixelizeImage": "JarJar Pixelize Image",
    "JarJarMakeShadow": "JarJar Make Shadow (Auto-Crop)",
}
