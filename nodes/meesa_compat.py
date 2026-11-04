"""Backward-compat shim for the retired monolithic ``MeesaJarJar`` node.

MeesaMultisMaker (``ComfyUIClient.ControlNetDepthWorkflow``) emits
``class_type == "MeesaJarJar"`` with 6 inputs
``{threshold, feather, channel_mode, despill, invert_mask, image}``
whose RGBA output feeds ``SplitImageWithAlpha`` downstream. That key no
longer exists in this pack (only ``JarJarRemoveBackground`` with
``{image, threshold}`` remains), so old saved workflows fail to load.

This module re-exports the old key and delegates to the existing
``JarJarRemoveBackground`` threshold logic where possible.

Limitations (documented, best-effort by design):
- ``channel_mode`` selects how the per-pixel background level is
  computed: ``rgb_max`` (max of R/G/B), ``luminance``
  (0.299R+0.587G+0.114B, identical to ``JarJarRemoveBackground``),
  or a single ``red``/``green``/``blue`` channel. The retired node
  may have used a different level metric.
- ``feather`` applies a PIL ``GaussianBlur(radius=feather)`` to the
  binary alpha mask (0 disables). Edge falloff curve of the retired
  node is unknown.
- ``despill`` is a best-effort fringe suppressor: edge RGB is lerped
  toward the pixel's own max channel by ``despill * (1 - alpha)`` to
  cut the dark halo left by keying. Strength 0 disables.
- ``invert_mask`` accepts ``"no"``/``"yes"`` (also bools/0/1 and
  "true"/"false"/"on"/"off" spellings); ``"yes"`` returns ``1-alpha``.
- If the input already carries an alpha channel, the computed mask is
  multiplied by it instead of discarding it.
"""

import numpy as np
import torch
from PIL import Image, ImageFilter

from .classic_remove_background import JarJarRemoveBackground

_CHANNEL_MODES = ["rgb_max", "luminance", "red", "green", "blue"]
_INVERT_MODES = ["no", "yes"]


class MeesaJarJar:
    """Legacy 6-input background remover -> RGBA IMAGE (compat shim)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "threshold": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "feather": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 64.0,
                    "step": 0.5,
                }),
                "channel_mode": (_CHANNEL_MODES, {"default": "rgb_max"}),
                "despill": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "invert_mask": (_INVERT_MODES, {"default": "no"}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "JarJar/Image"
    DESCRIPTION = (
        "Compat shim for the retired monolithic MeesaJarJar node "
        "(threshold/feather/channel_mode/despill/invert_mask -> RGBA). "
        "Threshold semantics match JarJarRemoveBackground; "
        "feather/despill/invert are best-effort (see module docstring)."
    )

    def _level(self, rgb: torch.Tensor, channel_mode: str) -> torch.Tensor:
        mode = str(channel_mode).lower()
        if mode == "luminance":
            return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if mode == "red":
            return rgb[..., 0]
        if mode == "green":
            return rgb[..., 1]
        if mode == "blue":
            return rgb[..., 2]
        # default: rgb_max
        return torch.amax(rgb, dim=-1)

    @staticmethod
    def _is_inverted(invert_mask) -> bool:
        if isinstance(invert_mask, bool):
            return invert_mask
        if isinstance(invert_mask, (int, float)):
            return bool(invert_mask)
        return str(invert_mask).strip().lower() in ("yes", "true", "1", "on", "invert")

    @staticmethod
    def _feather_alpha(alpha: torch.Tensor, feather: float) -> torch.Tensor:
        if feather is None or float(feather) <= 0:
            return alpha
        radius = float(feather)
        out = []
        for a in alpha.detach().cpu():
            pil = Image.fromarray((a.clamp(0, 1).numpy() * 255).astype("uint8"), mode="L")
            pil = pil.filter(ImageFilter.GaussianBlur(radius))
            t = torch.from_numpy(np.array(pil).astype("float32") / 255.0)
            out.append(t)
        feathered = torch.stack(out, dim=0).to(device=alpha.device, dtype=alpha.dtype)
        return feathered.clamp(0, 1)

    def process(self, threshold, feather, channel_mode, despill, invert_mask, image: torch.Tensor):
        rgb = image[..., :3]
        prior_alpha = image[..., 3] if image.shape[-1] == 4 else None

        # Fast path: exact JarJarRemoveBackground semantics.
        if (str(channel_mode).lower() == "luminance"
                and (feather is None or float(feather) <= 0)
                and (despill is None or float(despill) <= 0)
                and not self._is_inverted(invert_mask)
                and prior_alpha is None):
            return JarJarRemoveBackground().remove_bg(image, float(threshold))

        level = self._level(rgb, channel_mode)
        alpha = (level > float(threshold)).float()

        if self._is_inverted(invert_mask):
            alpha = 1.0 - alpha

        if prior_alpha is not None:
            alpha = alpha * prior_alpha.clamp(0, 1)

        alpha = self._feather_alpha(alpha, feather)

        s = 0.0 if despill is None else float(despill)
        if s > 0:
            s = max(0.0, min(1.0, s))
            k = (s * (1.0 - alpha)).unsqueeze(-1)  # (B,H,W,1)
            peak = torch.amax(rgb, dim=-1, keepdim=True)
            rgb = rgb * (1.0 - k) + peak * k

        out = torch.cat([rgb.clamp(0, 1), alpha.clamp(0, 1).unsqueeze(-1)], dim=-1)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MeesaJarJar": MeesaJarJar,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeesaJarJar": "Meesa JarJar (Legacy Compat)",
}
