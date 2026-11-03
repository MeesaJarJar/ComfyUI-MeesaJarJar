import torch
import numpy as np


class JarJarDither:
    """Apply ordered (Bayer) or Floyd-Steinberg dithering to create
    the classic retro pixel art look. Preserves alpha channel."""

    BAYER_4x4 = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5],
    ], dtype=np.float32) / 16.0

    BAYER_8x8 = np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ], dtype=np.float32) / 64.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dither_type": (["ordered_4x4", "ordered_8x8", "floyd_steinberg"],),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dither"
    CATEGORY = "JarJar/Image"

    def dither(self, image: torch.Tensor, dither_type: str, strength: float):
        B, H, W, C = image.shape
        has_alpha = C == 4
        result = []

        for b in range(B):
            img = (image[b].cpu().numpy() * 255).astype(np.float32)

            rgb = img[:, :, :3]
            alpha = img[:, :, 3:] if has_alpha else None

            if dither_type == "ordered_4x4":
                dithered = self._ordered_dither(rgb, self.BAYER_4x4)
            elif dither_type == "ordered_8x8":
                dithered = self._ordered_dither(rgb, self.BAYER_8x8)
            else:
                dithered = self._floyd_steinberg(rgb)

            if strength < 1.0:
                dithered = rgb * (1 - strength) + dithered * strength

            if has_alpha:
                out = np.concatenate([dithered, alpha], axis=2)
            else:
                out = dithered

            result.append(torch.from_numpy(out / 255.0).to(image.device))

        return (torch.stack(result, dim=0).clamp(0, 1),)

    def _ordered_dither(self, img: np.ndarray, bayer: np.ndarray) -> np.ndarray:
        h, w, c = img.shape
        bh, bw = bayer.shape
        output = img.copy()

        for y in range(h):
            for x in range(w):
                threshold = bayer[y % bh, x % bw]
                for ch in range(c):
                    val = output[y, x, ch]
                    output[y, x, ch] = 255.0 if val / 255.0 > threshold else 0.0

        return output

    def _floyd_steinberg(self, img: np.ndarray) -> np.ndarray:
        h, w, c = img.shape
        canvas = img.copy()

        for y in range(h):
            for x in range(w):
                old = canvas[y, x].copy()
                new = np.round(old / 255.0) * 255.0
                canvas[y, x] = new
                error = old - new
                if x + 1 < w:
                    canvas[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x - 1 >= 0:
                        canvas[y + 1, x - 1] += error * 3 / 16
                    canvas[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        canvas[y + 1, x + 1] += error * 1 / 16

        return canvas.clip(0, 255).astype(np.float32)
