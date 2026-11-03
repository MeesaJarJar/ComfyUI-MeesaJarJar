import torch
import numpy as np


class JarJarPaletteReduce:
    """Reduce image colors to a limited palette using k-means clustering.
    Mimics the limited color palettes used in Ultima Online sprites.
    Handles both RGB and RGBA images (preserves alpha)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 32,
                    "min": 2,
                    "max": 256,
                    "step": 1,
                    "display": "number",
                }),
                "dither": (["none", "floyd_steinberg"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reduce"
    CATEGORY = "JarJar/Image"

    def reduce(self, image: torch.Tensor, num_colors: int, dither: str):
        B, H, W, C = image.shape
        has_alpha = C == 4
        result = []

        for b in range(B):
            img = (image[b].cpu().numpy() * 255).astype(np.float32)

            rgb = img[:, :, :3]
            alpha = img[:, :, 3:] if has_alpha else None

            if dither == "floyd_steinberg":
                rgb = self._floyd_steinberg_dither(rgb, num_colors)
            else:
                rgb = self._quantize(rgb, num_colors)

            if has_alpha:
                out = np.concatenate([rgb, alpha], axis=2)
            else:
                out = rgb

            result.append(torch.from_numpy(out / 255.0).to(image.device))

        return (torch.stack(result, dim=0).clamp(0, 1),)

    def _quantize(self, img: np.ndarray, num_colors: int) -> np.ndarray:
        h, w, _ = img.shape
        pixels = img.reshape(-1, 3)

        flat = pixels.copy()
        for i in range(3):
            flat = flat[flat[:, i].argsort()]

        indices = np.linspace(0, len(flat) - 1, num_colors, dtype=int)
        centroids = flat[indices].copy()

        for _ in range(5):
            dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
            labels = dists.argmin(axis=1)
            for k in range(num_colors):
                mask = labels == k
                if mask.any():
                    centroids[k] = pixels[mask].mean(axis=0)

        dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = dists.argmin(axis=1)
        quantized = centroids[labels].reshape(h, w, 3)

        return quantized.astype(np.float32)

    def _floyd_steinberg_dither(self, img: np.ndarray, num_colors: int) -> np.ndarray:
        h, w, c = img.shape
        canvas = img.copy()

        for y in range(h):
            for x in range(w):
                old = canvas[y, x].copy()
                new = self._find_nearest(old, canvas.reshape(-1, 3), num_colors)
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

    def _find_nearest(self, pixel, palette, num_colors):
        dists = np.linalg.norm(palette - pixel, axis=1)
        return palette[dists.argmin()]
