"""JarJarSplatFillLight - lift baked-in darkness out of gaussian splats.

Single-view TripoSplat generations bake the input photo's lighting into the
splat SH colors: bright front, near-black back. This node adds a
direction-independent fill light to the SH *DC (diffuse) term only*,
leaving higher-order SH (view-dependent sheen) untouched:

  rgb = 0.5 + C0 * dc
  lift = fill * clamp((threshold - luminance) / threshold, 0, 1)
  rgb_new = clamp(rgb + lift, 0, 1)
  dc_new = (rgb_new - 0.5) / C0

Only splats darker than `threshold` move, proportionally to how dark they
are, so bright areas never blow out. Operates on the live SPLAT tensor
(no file roundtrip); wire between VAEDecodeTripoSplat and SplatToFile3D.
"""

import logging

import torch
from comfy_api.latest import IO

log = logging.getLogger("jarjar")

_SH_C0 = 0.28209479177387814


class JarJarSplatFillLight(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSplatFillLight",
            display_name="JarJar: Splat Fill Light",
            category="JarJar/3D",
            description=(
                "Fill light for gaussian splats: lifts splats darker than "
                "threshold toward fill brightness via the SH DC term. Fixes "
                "the dark-back-side look of single-view TripoSplat generations. "
                "Higher-order SH untouched."
            ),
            inputs=[
                IO.Splat.Input("splat"),
                IO.Float.Input(
                    "fill",
                    default=0.6,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Fill brightness added to fully-dark splats. "
                            "0 = off. Start ~0.5-0.8.",
                ),
                IO.Float.Input(
                    "threshold",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Luminance below which splats get lifted. "
                            "Only darker splats move.",
                ),
            ],
            outputs=[
                IO.Splat.Output(display_name="splat"),
            ],
        )

    @classmethod
    def execute(cls, splat, fill=0.6, threshold=0.3) -> IO.NodeOutput:
        from comfy_api.latest._util.geometry_types import SPLAT as SplatType

        if fill <= 0 or threshold <= 0:
            return IO.NodeOutput(splat)

        sh = splat.sh
        dc = sh[..., 0, :].float()  # (..., 3)
        rgb = 0.5 + _SH_C0 * dc
        lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
        t = max(float(threshold), 1e-6)
        w = torch.clamp((t - lum) / t, 0.0, 1.0)
        rgb_new = torch.clamp(rgb + float(fill) * w.unsqueeze(-1), 0.0, 1.0)
        dc_new = (rgb_new - 0.5) / _SH_C0

        sh_new = sh.clone()
        sh_new[..., 0, :] = dc_new.to(sh.dtype)
        n_dark = int((w > 0).sum())
        log.info("JarJarSplatFillLight: lifted %d splats (fill=%.2f threshold=%.2f).",
                 n_dark, fill, threshold)

        out = SplatType(
            splat.positions, splat.scales, splat.rotations,
            splat.opacities, sh_new, splat.counts,
        )
        return IO.NodeOutput(out)


NODE_CLASS_MAPPINGS = {
    "JarJarSplatFillLight": JarJarSplatFillLight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarSplatFillLight": "JarJar: Splat Fill Light",
}
