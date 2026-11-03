import torch
import numpy as np


class JarJarOutline:
    """Add black outlines around sprite edges and force full opacity.
    If input is RGBA, outlines based on alpha silhouette.
    If input is RGB, outlines based on color edges.
    Output is always RGBA with alpha=1 everywhere."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "outline_width": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "outline"
    CATEGORY = "JarJar/Image"

    def outline(self, image: torch.Tensor, outline_width: int):
        B, H, W, C = image.shape
        has_alpha = C == 4
        result = []

        for b in range(B):
            img = image[b].cpu().numpy()

            if has_alpha:
                alpha = img[:, :, 3]
                object_mask = alpha > 0.5
                edge_mask = self._dilate(~object_mask, outline_width) & object_mask
                # Force full opacity
                out_rgb = img[:, :, :3].copy()
                out_alpha = np.where(object_mask, 1.0, 0.0)
            else:
                gray = img.mean(axis=2)
                object_mask = gray > 0.01
                edge_mask = self._dilate(~object_mask, outline_width) & object_mask
                out_rgb = img.copy()
                out_alpha = np.ones((H, W), dtype=np.float32)

            # Draw black outline
            out_rgb[edge_mask] = [0.0, 0.0, 0.0]

            out = np.concatenate([
                out_rgb,
                out_alpha[:, :, np.newaxis],
            ], axis=2)

            result.append(torch.from_numpy(out).to(image.device))

        return (torch.stack(result, dim=0).clamp(0, 1),)

    def _dilate(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Dilate a boolean mask by expanding True pixels outward."""
        h, w = mask.shape
        result = mask.copy()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(mask.astype(np.uint8), -dy, axis=0), -dx, axis=1).astype(bool)
                result |= shifted
        return result
