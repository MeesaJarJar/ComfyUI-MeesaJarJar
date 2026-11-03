import torch
import numpy as np


class JarJarShadow:
    """Cast a drop shadow from a sprite. Shadow is anchored to the bottom-left
    (isometric diamond base) and skews outward. Auto-enlarges canvas.
    Outputs: image with shadow composited, and shadow-only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("FLOAT", {
                    "default": 135.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "slider",
                }),
                "distance": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "display": "number",
                }),
                "shadow_color": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "shadow_opacity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "skew": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "stretch": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "display": "slider",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "shadow")
    FUNCTION = "cast_shadow"
    CATEGORY = "JarJar/Image"

    def cast_shadow(self, image: torch.Tensor, angle: float, distance: int,
                    shadow_color: float, shadow_opacity: float,
                    skew: float, stretch: float):
        B, H, W, C = image.shape
        has_alpha = C == 4
        composited = []
        shadows = []

        rad = np.radians(angle)
        base_dx = np.cos(rad) * distance
        base_dy = np.sin(rad) * distance

        max_skew = int(abs(skew) * H) + int(abs(base_dx)) + 4
        max_vert = int(abs(base_dy) * stretch) + 4
        pad_top = max_vert + 2
        pad_bottom = max_vert + 2
        pad_left = max_skew + 2
        pad_right = max_skew + 2
        new_h = H + pad_top + pad_bottom
        new_w = W + pad_left + pad_right

        skew_sign = 1.0 if base_dx >= 0 else -1.0

        for b in range(B):
            img = image[b].cpu().numpy()

            if has_alpha:
                alpha = img[:, :, 3]
                rgb = img[:, :, :3]
            else:
                alpha = np.ones((H, W), dtype=np.float32)
                rgb = img

            object_mask = alpha > 0.5
            oy_coords, ox_coords = np.where(object_mask)

            shadow_mask = np.zeros((new_h, new_w), dtype=np.float32)

            if len(oy_coords) > 0:
                dist_from_bottom = (H - 1 - oy_coords).astype(np.float32)
                dist_from_left = ox_coords.astype(np.float32)
                skew_amount = skew_sign * skew * (dist_from_bottom + dist_from_left)

                dst_x = (pad_left + ox_coords + np.round(base_dx + skew_amount)).astype(np.int32)
                dst_y = (pad_top + oy_coords + np.round(base_dy * stretch)).astype(np.int32)

                valid = (dst_x >= 0) & (dst_x < new_w) & (dst_y >= 0) & (dst_y < new_h)
                shadow_mask[dst_y[valid], dst_x[valid]] = 1.0

            # Shadow image
            shadow_rgb = np.zeros((new_h, new_w, 3), dtype=np.float32)
            shadow_alpha = shadow_mask * shadow_opacity
            for ch in range(3):
                shadow_rgb[:, :, ch] = shadow_mask * shadow_color
            shadow_img = np.concatenate([shadow_rgb, shadow_alpha[:, :, np.newaxis]], axis=2)

            # Composite: shadow behind sprite
            sy1, sy2 = pad_top, pad_top + H
            sx1, sx2 = pad_left, pad_left + W
            comp_rgb = shadow_rgb.copy()
            comp_alpha = shadow_alpha.copy()

            sprite_3d = object_mask[:, :, np.newaxis]
            comp_rgb[sy1:sy2, sx1:sx2] = np.where(sprite_3d, rgb, comp_rgb[sy1:sy2, sx1:sx2])
            comp_alpha[sy1:sy2, sx1:sx2] = np.maximum(comp_alpha[sy1:sy2, sx1:sx2], alpha)

            comp_out = np.concatenate([comp_rgb, comp_alpha[:, :, np.newaxis]], axis=2)
            composited.append(torch.from_numpy(comp_out).to(image.device))
            shadows.append(torch.from_numpy(shadow_img).to(image.device))

        return (
            torch.stack(composited, dim=0).clamp(0, 1),
            torch.stack(shadows, dim=0).clamp(0, 1),
        )
