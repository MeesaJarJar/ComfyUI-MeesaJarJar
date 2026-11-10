"""Preview a 9-slice scaling of an image.

The source is cut into a 3x3 grid of nine equal thirds (no border size
is configurable). The four corners stay fixed, the top/bottom-middle tiles
repeat horizontally, the left/right-middle tiles repeat vertically, and the
center tile repeats in both directions.
"""

import torch


class JarJar9SliceViewer:
    """Assemble a 9-slice sample from nine equal image thirds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "repeat_width": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                }),
                "repeat_height": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "JarJar/Image"
    DESCRIPTION = (
        "Slice the image into nine equal thirds (3x3 grid). Corners are kept "
        "fixed once, top/bottom-middle tiles repeat horizontally, left/right-"
        "middle tiles repeat vertically, and the center repeats in both "
        "directions. repeat_width/repeat_height are total tile counts (min 1)."
    )

    def render(self, image, repeat_width, repeat_height):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        _, H, W, _ = image.shape
        rw = repeat_width
        rh = repeat_height

        # Equal thirds; any non-divisible remainder is absorbed by the middle
        # tile so the four corners always stay the same size.
        top_h = H // 3
        bottom_h = H // 3
        mid_h = H - top_h - bottom_h
        left_w = W // 3
        right_w = W // 3
        mid_w = W - left_w - right_w

        t = top_h
        b = top_h + mid_h
        l = left_w
        r = left_w + mid_w

        tl = image[:, :t, :l, :]
        tm = image[:, :t, l:r, :]
        tr = image[:, :t, r:, :]
        ml = image[:, t:b, :l, :]
        mm = image[:, t:b, l:r, :]
        mr = image[:, t:b, r:, :]
        bl = image[:, b:, :l, :]
        bm = image[:, b:, l:r, :]
        br = image[:, b:, r:, :]

        top_row = torch.cat((tl, tm.repeat(1, 1, rw, 1), tr), dim=2)
        mid_row = torch.cat(
            (ml.repeat(1, rh, 1, 1), mm.repeat(1, rh, rw, 1), mr.repeat(1, rh, 1, 1)),
            dim=2,
        )
        bot_row = torch.cat((bl, bm.repeat(1, 1, rw, 1), br), dim=2)

        out = torch.cat((top_row, mid_row, bot_row), dim=1)
        return (out.clamp(0, 1),)


NODE_CLASS_MAPPINGS = {
    "JarJar9SliceViewer": JarJar9SliceViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJar9SliceViewer": "JarJar 9 Slice Viewer",
}
