"""JarJar splat upscalers - densify gaussian splats at inference time."""

import logging

import numpy as np
import torch
from comfy_api.latest import IO
from PIL import Image
from scipy.spatial import cKDTree
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from comfy_api.latest._util.geometry_types import SPLAT as _SPLAT_TYPE
except ImportError:  # pragma: no cover
    from comfy_api.latest import Types as _T
    _SPLAT_TYPE = _T.SPLAT

try:
    from comfy_extras.nodes_gaussian_splat import RenderSplat as _RenderSplat
except ImportError:
    _RenderSplat = None


def _quat_to_mat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def _pad_stack(items, n):
    tail = items[0].shape[1:]
    out = items[0].new_zeros((len(items), n, *tail))
    for i, t in enumerate(items):
        out[i, :t.shape[0]] = t
    return out


_C0 = 0.28209479177387814


class JarJarUpscaleSplatsSubdivide(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarUpscaleSplatsSubdivide",
            display_name="JarJar: Upscale Splats (Subdivide)",
            category="JarJar/3D",
            description="Densify a gaussian splat by subdividing every gaussian into a grid of smaller gaussians "
                        "(factor^3 children). Preserves coverage and color while multiplying splat count.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Int.Input("factor", default=2, min=2, max=3, step=1,
                             tooltip="Subdivision factor per axis. 2 = 8 children per gaussian, 3 = 27."),
                IO.Float.Input("scale_decay", default=1.0, min=0.5, max=2.0, step=0.05, advanced=True,
                               tooltip="Multiplier applied to child scale (>1 softer, <1 sharper)."),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, factor, scale_decay) -> IO.NodeOutput:
        b = splat.positions.shape[0]
        pos_b, scl_b, rot_b, op_b, sh_b, lengths = [], [], [], [], [], []
        t = (torch.arange(factor, dtype=torch.float32) - (factor - 1) / 2.0) * (2.0 / factor)
        grid = torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=-1).reshape(-1, 3)
        Nc = grid.shape[0]
        for i in range(b):
            n = int(splat.counts[i]) if splat.counts is not None else splat.positions.shape[1]
            dev = splat.positions.device
            dt = splat.positions.dtype
            if n == 0:
                pos_b.append(torch.zeros((0, 3), dtype=dt, device=dev))
                scl_b.append(torch.zeros((0, 3), dtype=splat.scales.dtype, device=dev))
                rot_b.append(torch.zeros((0, 4), dtype=splat.rotations.dtype, device=dev))
                op_b.append(torch.zeros((0, 1), dtype=splat.opacities.dtype, device=dev))
                sh_b.append(torch.zeros((0, splat.sh.shape[2], 3), dtype=splat.sh.dtype, device=dev))
                lengths.append(0)
                continue
            pos = splat.positions[i, :n]
            scl = splat.scales[i, :n]
            rot = splat.rotations[i, :n]
            op = splat.opacities[i, :n]
            sh = splat.sh[i, :n]
            grid_d = grid.to(device=dev, dtype=dt)
            R = _quat_to_mat(rot)
            child_off = grid_d[None, :, :] * scl[:, None, :]
            child_off_w = torch.einsum("nij,ncj->nci", R, child_off)
            child_pos = (pos[:, None, :] + child_off_w).reshape(-1, 3)
            child_scl = (scl[:, None, :] / float(factor) * float(scale_decay)).reshape(-1, 3).clamp_min(1e-6)
            child_rot = rot[:, None, :].expand(-1, Nc, -1).reshape(-1, 4)
            child_op = (op[:, None, :] / float(Nc)).reshape(-1, 1)
            child_sh = sh[:, None, :, :].expand(-1, Nc, -1, -1).reshape(-1, sh.shape[1], 3)
            pos_b.append(child_pos); scl_b.append(child_scl); rot_b.append(child_rot)
            op_b.append(child_op); sh_b.append(child_sh)
            lengths.append(child_pos.shape[0])
        nmax = max(1, max(lengths) if lengths else 1)
        pv = _pad_stack(pos_b, nmax)
        sv = _pad_stack(scl_b, nmax)
        rv = _pad_stack(rot_b, nmax)
        ov = _pad_stack(op_b, nmax)
        shv = _pad_stack(sh_b, nmax)
        counts = torch.tensor(lengths, device=pv.device, dtype=torch.int64)
        uniform = bool((counts == nmax).all())
        out = _SPLAT_TYPE(pv, sv, rv, ov, shv, counts=None if uniform else counts)
        return IO.NodeOutput(out)


class JarJarUpscaleSplatsAdaptive(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarUpscaleSplatsAdaptive",
            display_name="JarJar: Upscale Splats (Adaptive Detail)",
            category="JarJar/3D",
            search_aliases=["upscale splat", "densify gaussian", "add detail splat"],
            description="Densify a gaussian splat adaptively: gaussians in high color-variance regions are split "
                        "along their longest axis with colors pushed toward local neighbors, injecting detail where "
                        "the splat is coarse. Low-variance gaussians are kept unchanged.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Int.Input("neighbors", default=8, min=4, max=32, step=1, advanced=True,
                             tooltip="k nearest neighbors used to estimate local color variance."),
                IO.Float.Input("variance_threshold", default=0.005, min=0.0, max=1.0, step=0.001,
                               tooltip="Split gaussians whose local color variance exceeds this. Lower = more splits."),
                IO.Float.Input("color_push", default=0.5, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip="How far child colors move toward the neighbor mean."),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, neighbors, variance_threshold, color_push) -> IO.NodeOutput:
        b = splat.positions.shape[0]
        pos_b, scl_b, rot_b, op_b, sh_b, lengths = [], [], [], [], [], []
        for i in range(b):
            n = int(splat.counts[i]) if splat.counts is not None else splat.positions.shape[1]
            dev = splat.positions.device
            dt = splat.positions.dtype
            if n == 0:
                pos_b.append(torch.zeros((0, 3), dtype=dt, device=dev))
                scl_b.append(torch.zeros((0, 3), dtype=splat.scales.dtype, device=dev))
                rot_b.append(torch.zeros((0, 4), dtype=splat.rotations.dtype, device=dev))
                op_b.append(torch.zeros((0, 1), dtype=splat.opacities.dtype, device=dev))
                sh_b.append(torch.zeros((0, splat.sh.shape[2], 3), dtype=splat.sh.dtype, device=dev))
                lengths.append(0)
                continue
            pos = splat.positions[i, :n].detach().cpu().numpy().astype("float32")
            scl = splat.scales[i, :n]
            rot = splat.rotations[i, :n]
            op = splat.opacities[i, :n]
            sh = splat.sh[i, :n]
            K = sh.shape[1]
            kq = min(neighbors + 1, n)
            tree = cKDTree(pos)
            _, nbr_idx = tree.query(pos, k=kq)
            if nbr_idx.ndim == 1:
                nbr_idx = nbr_idx[:, None]
            dc_all = sh[:, 0, :].float()
            rgb_all = 0.5 + _C0 * dc_all
            nbr_rgb = rgb_all[nbr_idx]
            var = nbr_rgb.std(dim=1).mean(dim=-1)
            split_mask = (var > variance_threshold).to(dev)
            out_pos = []; out_scl = []; out_rot = []; out_op = []; out_sh = []
            Rmat = _quat_to_mat(rot)
            for g in range(n):
                if not bool(split_mask[g].item()):
                    out_pos.append(splat.positions[i, g].unsqueeze(0).to(dev))
                    out_scl.append(scl[g].unsqueeze(0))
                    out_rot.append(rot[g].unsqueeze(0))
                    out_op.append(op[g].unsqueeze(0))
                    out_sh.append(sh[g].unsqueeze(0))
                    continue
                axis = int(scl[g].argmax().item())
                axis_dir = Rmat[g, :, axis]
                half = float(scl[g, axis].item()) * 0.5
                off = (axis_dir * half).to(device=dev, dtype=dt)
                p = splat.positions[i, g].to(dev)
                pos1 = (p + off).unsqueeze(0)
                pos2 = (p - off).unsqueeze(0)
                scl1 = scl[g].clone(); scl1[axis] = scl[g, axis] * 0.5
                scl2 = scl1.clone()
                rgb_p = rgb_all[g]
                nbr_mean = nbr_rgb[g, 1:].mean(dim=0) if kq > 1 else rgb_p
                delta = (nbr_mean - rgb_p) * float(color_push)
                rgb1 = (rgb_p + delta).clamp(0.0, 1.0)
                rgb2 = (rgb_p - delta).clamp(0.0, 1.0)
                dc1 = ((rgb1 - 0.5) / _C0).to(sh.dtype)
                dc2 = ((rgb2 - 0.5) / _C0).to(sh.dtype)
                sh1 = sh[g].clone(); sh1[0, :] = dc1
                sh2 = sh[g].clone(); sh2[0, :] = dc2
                op_half = op[g] * 0.5
                out_pos.append(pos1); out_pos.append(pos2)
                out_scl.append(scl1.unsqueeze(0)); out_scl.append(scl2.unsqueeze(0))
                out_rot.append(rot[g].unsqueeze(0)); out_rot.append(rot[g].unsqueeze(0))
                out_op.append(op_half.unsqueeze(0)); out_op.append(op_half.unsqueeze(0))
                out_sh.append(sh1.unsqueeze(0)); out_sh.append(sh2.unsqueeze(0))
            if len(out_pos) == 0:
                pos_b.append(torch.zeros((0, 3), dtype=dt, device=dev))
                scl_b.append(torch.zeros((0, 3), dtype=splat.scales.dtype, device=dev))
                rot_b.append(torch.zeros((0, 4), dtype=splat.rotations.dtype, device=dev))
                op_b.append(torch.zeros((0, 1), dtype=splat.opacities.dtype, device=dev))
                sh_b.append(torch.zeros((0, K, 3), dtype=splat.sh.dtype, device=dev))
                lengths.append(0)
                continue
            pos_b.append(torch.cat(out_pos, dim=0).to(dev))
            scl_b.append(torch.cat(out_scl, dim=0).to(dev))
            rot_b.append(torch.cat(out_rot, dim=0).to(dev))
            op_b.append(torch.cat(out_op, dim=0).to(dev))
            sh_b.append(torch.cat(out_sh, dim=0).to(dev))
            lengths.append(pos_b[-1].shape[0])
        nmax = max(1, max(lengths) if lengths else 1)
        pv = _pad_stack(pos_b, nmax)
        sv = _pad_stack(scl_b, nmax)
        rv = _pad_stack(rot_b, nmax)
        ov = _pad_stack(op_b, nmax)
        shv = _pad_stack(sh_b, nmax)
        counts = torch.tensor(lengths, device=pv.device, dtype=torch.int64)
        uniform = bool((counts == nmax).all())
        out = _SPLAT_TYPE(pv, sv, rv, ov, shv, counts=None if uniform else counts)
        return IO.NodeOutput(out)



class JarJarDownsampleSplatsTopK(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarDownsampleSplatsTopK",
            display_name="JarJar: Downsample Splats (Top-K)",
            category="JarJar/3D",
            search_aliases=["downsample splat", "reduce gaussians", "prune splat", "splat simplify"],
            description="Downsample a gaussian splat to a target count by keeping the most important gaussians "
                        "(opacity * scale volume). Fast and preserves salient detail.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Int.Input("target_count", default=100000, min=1000, max=1000000, step=1000,
                             tooltip="Desired number of gaussians after downsampling."),
                IO.Float.Input("opacity_floor", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Drop gaussians below this opacity before selecting (0 = keep all)."),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, target_count, opacity_floor) -> IO.NodeOutput:
        b = splat.positions.shape[0]
        pos_b, scl_b, rot_b, op_b, sh_b, lengths = [], [], [], [], [], []
        for i in range(b):
            n = int(splat.counts[i]) if splat.counts is not None else splat.positions.shape[1]
            dev = splat.positions.device
            dt = splat.positions.dtype
            if n == 0:
                pos_b.append(torch.zeros((0, 3), dtype=dt, device=dev))
                scl_b.append(torch.zeros((0, 3), dtype=splat.scales.dtype, device=dev))
                rot_b.append(torch.zeros((0, 4), dtype=splat.rotations.dtype, device=dev))
                op_b.append(torch.zeros((0, 1), dtype=splat.opacities.dtype, device=dev))
                sh_b.append(torch.zeros((0, splat.sh.shape[2], 3), dtype=splat.sh.dtype, device=dev))
                lengths.append(0)
                continue
            pos = splat.positions[i, :n]
            scl = splat.scales[i, :n]
            rot = splat.rotations[i, :n]
            op = splat.opacities[i, :n]
            sh = splat.sh[i, :n]
            if opacity_floor > 0:
                keep0 = op[:, 0] >= opacity_floor
                pos = pos[keep0]; scl = scl[keep0]; rot = rot[keep0]; op = op[keep0]; sh = sh[keep0]
                n = pos.shape[0]
            if n == 0:
                pos_b.append(torch.zeros((0, 3), dtype=dt, device=dev))
                scl_b.append(torch.zeros((0, 3), dtype=splat.scales.dtype, device=dev))
                rot_b.append(torch.zeros((0, 4), dtype=splat.rotations.dtype, device=dev))
                op_b.append(torch.zeros((0, 1), dtype=splat.opacities.dtype, device=dev))
                sh_b.append(torch.zeros((0, splat.sh.shape[2], 3), dtype=splat.sh.dtype, device=dev))
                lengths.append(0)
                continue
            importance = op[:, 0].float() * scl.prod(dim=-1).float().clamp_min(1e-12)
            k = min(int(target_count), n)
            idx = torch.topk(importance, k).indices
            pos_b.append(pos[idx]); scl_b.append(scl[idx]); rot_b.append(rot[idx])
            op_b.append(op[idx]); sh_b.append(sh[idx])
            lengths.append(k)
        nmax = max(1, max(lengths) if lengths else 1)
        pv = _pad_stack(pos_b, nmax)
        sv = _pad_stack(scl_b, nmax)
        rv = _pad_stack(rot_b, nmax)
        ov = _pad_stack(op_b, nmax)
        shv = _pad_stack(sh_b, nmax)
        counts = torch.tensor(lengths, device=pv.device, dtype=torch.int64)
        uniform = bool((counts == nmax).all())
        out = _SPLAT_TYPE(pv, sv, rv, ov, shv, counts=None if uniform else counts)
        return IO.NodeOutput(out)


class JarJarDownsampleSplatsVoxel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarDownsampleSplatsVoxel",
            display_name="JarJar: Downsample Splats (Voxel Merge)",
            category="JarJar/3D",
            search_aliases=["downsample splat", "voxel merge splat", "splat simplify", "merge gaussians"],
            description="Downsample a gaussian splat by clustering gaussians into a voxel grid and merging each "
                        "cluster into a single gaussian (opacity-weighted). Smooth, topology-preserving reduction.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Int.Input("grid", default=64, min=8, max=512, step=8,
                             tooltip="Voxel grid resolution along the longest axis. Higher = more gaussians kept."),
                IO.Float.Input("opacity_floor", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Drop gaussians below this opacity before merging."),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, grid, opacity_floor) -> IO.NodeOutput:
        b = splat.positions.shape[0]
        pos_b, scl_b, rot_b, op_b, sh_b, lengths = [], [], [], [], [], []
        for i in range(b):
            n = int(splat.counts[i]) if splat.counts is not None else splat.positions.shape[1]
            dev = splat.positions.device
            dt = splat.positions.dtype
            if n == 0:
                pos_b.append(torch.zeros((0, 3), dtype=dt, device=dev))
                scl_b.append(torch.zeros((0, 3), dtype=splat.scales.dtype, device=dev))
                rot_b.append(torch.zeros((0, 4), dtype=splat.rotations.dtype, device=dev))
                op_b.append(torch.zeros((0, 1), dtype=splat.opacities.dtype, device=dev))
                sh_b.append(torch.zeros((0, splat.sh.shape[2], 3), dtype=splat.sh.dtype, device=dev))
                lengths.append(0)
                continue
            pos = splat.positions[i, :n]
            scl = splat.scales[i, :n]
            rot = splat.rotations[i, :n]
            op = splat.opacities[i, :n]
            sh = splat.sh[i, :n]
            K = sh.shape[1]
            if opacity_floor > 0:
                keep0 = op[:, 0] >= opacity_floor
                pos = pos[keep0]; scl = scl[keep0]; rot = rot[keep0]; op = op[keep0]; sh = sh[keep0]
                n = pos.shape[0]
            if n == 0:
                pos_b.append(torch.zeros((0, 3), dtype=dt, device=dev))
                scl_b.append(torch.zeros((0, 3), dtype=splat.scales.dtype, device=dev))
                rot_b.append(torch.zeros((0, 4), dtype=splat.rotations.dtype, device=dev))
                op_b.append(torch.zeros((0, 1), dtype=splat.opacities.dtype, device=dev))
                sh_b.append(torch.zeros((0, K, 3), dtype=splat.sh.dtype, device=dev))
                lengths.append(0)
                continue
            # voxel coords
            lo = pos.min(0).values
            hi = pos.max(0).values
            extent = (hi - lo).clamp_min(1e-6)
            voxel = extent.max() / float(grid)
            coords = ((pos - lo) / voxel).floor().long()
            # unique cell keys
            key = coords[:, 0] * (grid + 1) * (grid + 1) + coords[:, 1] * (grid + 1) + coords[:, 2]
            uniq, inv = torch.unique(key, return_inverse=True)
            ncells = uniq.shape[0]
            # accumulate
            w = op[:, 0].clamp_min(1e-8).float()
            # weighted sums
            pos_sum = torch.zeros((ncells, 3), dtype=torch.float32, device=dev)
            sh_sum = torch.zeros((ncells, K, 3), dtype=torch.float32, device=dev)
            w_sum = torch.zeros((ncells,), dtype=torch.float32, device=dev)
            scl_sq_sum = torch.zeros((ncells, 3), dtype=torch.float32, device=dev)
            pos_sum.index_add_(0, inv, (pos.float() * w[:, None]))
            sh_sum.index_add_(0, inv, (sh.float() * w[:, None, None]))
            w_sum.index_add_(0, inv, w)
            scl_sq_sum.index_add_(0, inv, (scl.float().square() * w[:, None]))
            # positions (pos weighted) / w
            new_pos = pos_sum / w_sum[:, None]
            new_sh = sh_sum / w_sum[:, None, None]
            # scale: sqrt( weighted var of positions + mean scl^2 )
            # compute pos^2 weighted sum
            pos_sq_sum = torch.zeros((ncells, 3), dtype=torch.float32, device=dev)
            pos_sq_sum.index_add_(0, inv, (pos.float().square() * w[:, None]))
            var = (pos_sq_sum / w_sum[:, None]) - new_pos.square()
            var = var.clamp_min(0.0)
            mean_scl_sq = scl_sq_sum / w_sum[:, None]
            new_scl = (var + mean_scl_sq).sqrt().clamp_min(1e-4)
            new_op = w_sum.clamp(0.0, 1.0).unsqueeze(-1)
            new_rot = torch.tensor([1., 0, 0, 0], device=dev, dtype=dt).unsqueeze(0).expand(ncells, -1).clone()
            pos_b.append(new_pos.to(dt))
            scl_b.append(new_scl.to(splat.scales.dtype))
            rot_b.append(new_rot)
            op_b.append(new_op.to(splat.opacities.dtype))
            sh_b.append(new_sh.to(splat.sh.dtype))
            lengths.append(ncells)
        nmax = max(1, max(lengths) if lengths else 1)
        pv = _pad_stack(pos_b, nmax)
        sv = _pad_stack(scl_b, nmax)
        rv = _pad_stack(rot_b, nmax)
        ov = _pad_stack(op_b, nmax)
        shv = _pad_stack(sh_b, nmax)
        counts = torch.tensor(lengths, device=pv.device, dtype=torch.int64)
        uniform = bool((counts == nmax).all())
        out = _SPLAT_TYPE(pv, sv, rv, ov, shv, counts=None if uniform else counts)
        return IO.NodeOutput(out)


class JarJarSplatReconstructScore(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSplatReconstructScore",
            display_name="JarJar: Splat Reconstruction Score",
            category="JarJar/3D",
            description="Render a splat to an image (auto-framed) and compare to a reference image, reporting "
                        "PSNR/SSIM/MSE. Embeds the render+compare step into one node for GAN-style SNR scoring.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Image.Input("reference_image"),
                IO.Int.Input("width", default=512, min=64, max=2048, step=8),
                IO.Int.Input("height", default=512, min=64, max=2048, step=8),
            ],
            outputs=[
                IO.Image.Output(display_name="rendered"),
                IO.Float.Output(display_name="psnr"),
                IO.Float.Output(display_name="ssim"),
                IO.Float.Output(display_name="mse"),
            ],
        )

    @classmethod
    def execute(cls, splat, reference_image, width, height) -> IO.NodeOutput:
        if _RenderSplat is None:
            raise RuntimeError("RenderSplat not available")
        res = _RenderSplat.execute(splat, width=width, height=height, frames=1,
            splat_scale=1.0, sharpen=0.0, headlight_shading=False,
            opacity_threshold=0.0, background="#000000", render_style="all",
            camera_info=None)
        rendered = res[0] if not hasattr(res, "outputs") else res.outputs[0]
        if isinstance(rendered, (tuple, list)):
            rendered = rendered[0]
        rendered = rendered[0] if rendered.dim() == 4 else rendered
        a = (rendered.detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
        ref = reference_image[0] if reference_image.dim() == 4 else reference_image
        b = (ref.detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
        if a.shape != b.shape:
            bb = Image.fromarray(b)
            bb = bb.resize((a.shape[1], a.shape[0]), Image.BILINEAR)
            b = np.asarray(bb)
        psnr = float(peak_signal_noise_ratio(a, b, data_range=255))
        ssim = float(structural_similarity(a, b, channel_axis=-1, data_range=255))
        mse = float(((a.astype("float32") - b.astype("float32")) ** 2).mean())
        rendered_out = rendered.detach().cpu().unsqueeze(0) if rendered.dim() == 3 else rendered.detach().cpu()
        return IO.NodeOutput(rendered_out, psnr, ssim, mse)


class JarJarSplatEnhance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JarJarSplatEnhance",
            display_name="JarJar: Splat Enhance",
            category="JarJar/3D",
            search_aliases=["enhance splat", "contrast splat", "saturation splat", "sharpen splat"],
            description="Enhance a gaussian splat's colors: brightness, contrast, saturation, levels, and "
                        "view-dependent SH boost. Operates on the SH DC term (base color) and higher-order SH.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Float.Input("brightness", default=0.0, min=-1.0, max=1.0, step=0.01,
                               tooltip="Additive brightness shift on base color."),
                IO.Float.Input("contrast", default=1.0, min=0.0, max=4.0, step=0.05,
                               tooltip="Contrast around 0.5 (1.0 = unchanged)."),
                IO.Float.Input("saturation", default=1.0, min=0.0, max=4.0, step=0.05,
                               tooltip="Saturation (0 = grayscale, 1 = unchanged)."),
                IO.Float.Input("in_black", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Levels input black point."),
                IO.Float.Input("in_white", default=1.0, min=0.01, max=1.0, step=0.01, advanced=True,
                               tooltip="Levels input white point."),
                IO.Float.Input("gamma", default=1.0, min=0.1, max=3.0, step=0.05, advanced=True,
                               tooltip="Levels gamma."),
                IO.Float.Input("sh_boost", default=1.0, min=0.0, max=4.0, step=0.05, advanced=True,
                               tooltip="Multiplier on higher-order SH (view-dependent detail). 1 = unchanged."),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, brightness, contrast, saturation, in_black, in_white, gamma, sh_boost) -> IO.NodeOutput:
        sh = splat.sh.clone()
        dc = sh[..., 0, :].float()  # (...,K? no) shape (B,N,3) if batched; here per-tensor
        # sh shape (B,N,K,3) for batched; handle generically
        rgb = 0.5 + _C0 * dc  # (..., 3)
        # levels
        denom = max(float(in_white) - float(in_black), 1e-6)
        rgb = (rgb - float(in_black)) / denom
        rgb = rgb.clamp(0.0, 1.0)
        if gamma != 1.0:
            rgb = rgb.clamp(0.0, 1.0).pow(1.0 / float(gamma))
        # contrast
        rgb = (rgb - 0.5) * float(contrast) + 0.5
        # brightness
        rgb = rgb + float(brightness)
        # saturation
        if saturation != 1.0:
            gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).unsqueeze(-1)
            rgb = gray + (rgb - gray) * float(saturation)
        rgb = rgb.clamp(0.0, 1.0)
        dc_new = (rgb - 0.5) / _C0
        sh[..., 0, :] = dc_new.to(sh.dtype)
        if sh_boost != 1.0 and sh.shape[2] > 1:
            sh[..., 1:, :] = (sh[..., 1:, :].float() * float(sh_boost)).to(sh.dtype)
        # rebuild splat preserving counts
        out = _SPLAT_TYPE(splat.positions, splat.scales, splat.rotations, splat.opacities, sh, splat.counts)
        return IO.NodeOutput(out)


NODE_CLASS_MAPPINGS = {
    "JarJarUpscaleSplatsSubdivide": JarJarUpscaleSplatsSubdivide,
    "JarJarUpscaleSplatsAdaptive": JarJarUpscaleSplatsAdaptive,
    "JarJarDownsampleSplatsTopK": JarJarDownsampleSplatsTopK,
    "JarJarDownsampleSplatsVoxel": JarJarDownsampleSplatsVoxel,
    "JarJarSplatReconstructScore": JarJarSplatReconstructScore,
    "JarJarSplatEnhance": JarJarSplatEnhance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarUpscaleSplatsSubdivide": "JarJar: Upscale Splats (Subdivide)",
    "JarJarUpscaleSplatsAdaptive": "JarJar: Upscale Splats (Adaptive Detail)",
    "JarJarDownsampleSplatsTopK": "JarJar: Downsample Splats (Top-K)",
    "JarJarDownsampleSplatsVoxel": "JarJar: Downsample Splats (Voxel Merge)",
    "JarJarSplatReconstructScore": "JarJar: Splat Reconstruction Score",
    "JarJarSplatEnhance": "JarJar: Splat Enhance",
}
