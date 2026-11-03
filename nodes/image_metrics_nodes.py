"""Image quality metrics nodes (PSNR/SSIM/MSE) for JarJar."""

import numpy as np
import torch
from comfy_api.latest import IO
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_uint8_img(t):
    # t: (H,W,3) or (B,H,W,3) float [0,1] or uint8
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu()
    if t.dtype != torch.uint8:
        t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return t.numpy()


class JarJarCompareImages(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarCompareImages",
            display_name="JarJar: Compare Images (PSNR/SSIM)",
            category="JarJar/3D",
            description="Compare two images and report PSNR, SSIM and MSE. Use after Render Splat to score "
                        "a splat's reconstruction against the source image.",
            inputs=[
                IO.Image.Input("image_a"),
                IO.Image.Input("image_b"),
            ],
            outputs=[
                IO.Float.Output(display_name="psnr"),
                IO.Float.Output(display_name="ssim"),
                IO.Float.Output(display_name="mse"),
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b) -> IO.NodeOutput:
        a = _to_uint8_img(image_a)
        b = _to_uint8_img(image_b)
        # resize b to a if needed
        if a.shape != b.shape:
            ba = Image.fromarray(b)
            ba = ba.resize((a.shape[1], a.shape[0]), Image.BILINEAR)
            b = np.asarray(ba)
        psnr = float(peak_signal_noise_ratio(a, b, data_range=255))
        ssim = float(structural_similarity(a, b, channel_axis=-1, data_range=255))
        mse = float(((a.astype("float32") - b.astype("float32")) ** 2).mean())
        return IO.NodeOutput(psnr, ssim, mse)


NODE_CLASS_MAPPINGS = {
    "JarJarCompareImages": JarJarCompareImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarCompareImages": "JarJar: Compare Images (PSNR/SSIM)",
}
