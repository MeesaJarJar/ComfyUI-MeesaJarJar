import torch
import numpy as np
from scipy.ndimage import label, binary_opening
from PIL import Image, ImageFilter


class JarJarAutoSegmentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "background": (["auto", "white", "black"], {"default": "auto"}),
                "threshold": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 254.0, "step": 1.0}),
                "min_area": ("INT", {"default": 500, "min": 1, "max": 10000000, "step": 1}),
                "padding": ("INT", {"default": 4, "min": 0, "max": 512, "step": 1}),
                "max_sprites": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "sort_order": (["left-to-right", "top-to-bottom", "big-to-small", "small-to-big"], {"default": "left-to-right"}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("sprites", "masks", "count")
    FUNCTION = "segment"
    CATEGORY = "JarJar/Image"

    def segment(self, image, background="auto", threshold=15.0, min_area=500, padding=4, max_sprites=64, sort_order="left-to-right", feather=0):
        threshold, min_area, padding, max_sprites, feather = float(threshold), int(min_area), int(padding), max(1, int(max_sprites)), int(feather)
        rgba_list, mask_list = [], []
        for b in range(image.shape[0]):
            rgb = (image[b][..., :3].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            H, W, _ = rgb.shape
            if background == "white":
                bg = np.array([255, 255, 255], dtype=np.int16)
            elif background == "black":
                bg = np.array([0, 0, 0], dtype=np.int16)
            else:
                py, px = min(5, H), min(5, W)
                corners = np.concatenate([rgb[:py, :px].reshape(-1, 3), rgb[:py, W - px:].reshape(-1, 3), rgb[H - py:, :px].reshape(-1, 3), rgb[H - py:, W - px:].reshape(-1, 3)], axis=0).astype(np.float32)
                bg = np.array([255, 255, 255], dtype=np.int16) if float(corners.mean()) > 128.0 else np.array([0, 0, 0], dtype=np.int16)
            fg = binary_opening(np.abs(rgb.astype(np.int16) - bg.reshape(1, 1, 3)).max(axis=2) > threshold, structure=np.ones((3, 3)))
            if not bool(fg.any()):
                continue
            lab, n = label(fg, structure=np.ones((3, 3)))
            comps = []
            for i in range(1, n + 1):
                ys, xs = np.nonzero(lab == i)
                if ys.size == 0:
                    continue
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                if int(ys.size) >= min_area:
                    comps.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1, int(ys.size), i))
            if sort_order == "left-to-right":
                comps.sort(key=lambda c: (c[0], c[1]))
            elif sort_order == "top-to-bottom":
                comps.sort(key=lambda c: (c[1], c[0]))
            elif sort_order == "big-to-small":
                comps.sort(key=lambda c: -c[4])
            elif sort_order == "small-to-big":
                comps.sort(key=lambda c: c[4])
            for x, y, w, h, _, cid in comps[:max_sprites]:
                x0, y0, x1, y1 = max(0, x - padding), max(0, y - padding), min(W, x + w + padding), min(H, y + h + padding)
                m = (lab[y0:y1, x0:x1] == cid).astype(np.float32)
                if feather > 0:
                    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(feather)), dtype=np.float32) / 255.0
                crop = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
                rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.float32)
                rgba[..., :3] = crop
                rgba[..., 3] = m
                rgba_list.append(torch.from_numpy(rgba))
                mask_list.append(torch.from_numpy(m))
        if not rgba_list:
            return (torch.zeros((1, 64, 64, 4), dtype=torch.float32), torch.zeros((1, 64, 64), dtype=torch.float32), 0)
        maxH = max(t.shape[0] for t in rgba_list)
        maxW = max(t.shape[1] for t in rgba_list)
        sprites, masks = [], []
        for r, m in zip(rgba_list, mask_list):
            hh, ww = r.shape[0], r.shape[1]
            canvas = torch.zeros((maxH, maxW, 4), dtype=torch.float32)
            mm = torch.zeros((maxH, maxW), dtype=torch.float32)
            canvas[:hh, :ww] = r
            mm[:hh, :ww] = m
            sprites.append(canvas)
            masks.append(mm)
        return (torch.stack(sprites, dim=0), torch.stack(masks, dim=0), len(sprites))


NODE_CLASS_MAPPINGS = {"JarJarAutoSegmentImage": JarJarAutoSegmentImage}
NODE_DISPLAY_NAME_MAPPINGS = {"JarJarAutoSegmentImage": "JarJar Auto Segment Image"}
