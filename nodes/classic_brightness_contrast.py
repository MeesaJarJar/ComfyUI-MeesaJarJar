import torch


class JarJarBrightnessContrast:
    """Adjust brightness and contrast. Useful for pushing realistic images
    toward the flat, punchy look of UO sprites."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "saturation": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "JarJar/Image"

    def adjust(self, image: torch.Tensor, brightness: float, contrast: float, saturation: float):
        x = image
        if brightness != 0:
            x = x + brightness
            x = x.clamp(0, 1)
        if contrast != 0:
            mid = 0.5
            factor = 1.0 + contrast
            x = (x - mid) * factor + mid
            x = x.clamp(0, 1)
        if saturation != 0:
            gray = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
            factor = 1.0 + saturation
            x = gray + factor * (x - gray)
            x = x.clamp(0, 1)
        return (x,)
