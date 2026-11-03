import torch


class JarJarPixelate:
    """Pixelate an image into sharp square blocks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pixelate"
    CATEGORY = "JarJar/Image"

    def pixelate(self, image: torch.Tensor, pixel_size: int):
        B, H, W, C = image.shape
        if pixel_size <= 1:
            return (image,)

        small_h = max(1, H // pixel_size)
        small_w = max(1, W // pixel_size)
        x = image.permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(
            x, size=(small_h, small_w), mode="nearest"
        )
        x = torch.nn.functional.interpolate(
            x, size=(H, W), mode="nearest"
        )
        return (x.permute(0, 2, 3, 1).clamp(0, 1),)
