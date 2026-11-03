import torch


class JarJarRemoveBackground:
    """Remove black background by converting dark pixels to transparent.
    Outputs RGBA image (4 channels)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_bg"
    CATEGORY = "JarJar/Image"

    def remove_bg(self, image: torch.Tensor, threshold: float):
        B, H, W, C = image.shape
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        alpha = (gray > threshold).float()
        out = torch.cat([image[..., :3], alpha.unsqueeze(-1)], dim=-1)
        return (out,)
