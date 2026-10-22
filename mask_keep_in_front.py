import torch
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask


class MaskKeepInFront:
    """Оставляет из маски только части (связные области), которые по карте
    глубины находятся ПЕРЕД объектом reference."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "reference": ("MASK",),
            "depth": ("IMAGE",),
            "depth_margin": ("FLOAT", {
                "default": 0.03, "min": 0.0, "max": 0.5, "step": 0.01,
                "tooltip": "Насколько область должна быть ближе переднего края "
                           "объекта reference, чтобы считаться передним планом."}),
            "min_area": ("INT", {
                "default": 64, "min": 0, "max": 100000, "step": 1,
                "tooltip": "Области меньше этой площади (px) отбрасываются."}),
            "depth_near_is_bright": ("BOOLEAN", {"default": True}),
        }}

    def run(self, mask, reference, depth, depth_margin, min_area, depth_near_is_bright):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        ref = reference[0].float().to(m.device)
        if tuple(ref.shape) != (H, W):
            ref = _resize_mask(ref, H, W)
        dep = depth[0].float().to(m.device)
        if dep.dim() == 3:
            dep = dep.mean(dim=-1)
        if tuple(dep.shape) != (H, W):
            dep = _resize_mask(dep, H, W)
        if not depth_near_is_bright:
            dep = 1.0 - dep

        m_np = (m > 0.5).cpu().numpy()
        r_np = (ref > 0.5).cpu().numpy()
        d_np = dep.cpu().numpy()
        out = np.zeros((H, W), dtype=np.float32)
        if r_np.any() and m_np.any():
            thr = float(np.quantile(d_np[r_np], 0.9)) + float(depth_margin)
            lab, n = ndi.label(m_np)
            kept = 0
            for i in range(1, n + 1):
                comp = lab == i
                if int(comp.sum()) < min_area:
                    continue
                if float(np.median(d_np[comp])) > thr:
                    out[comp] = 1.0
                    kept += 1
            info = (f"областей: {n}; оставлено (перед объектом): {kept}; "
                    f"порог глубины: {thr:.3f}")
        else:
            info = "маска или reference пустые — ничего не оставлено"
        return (torch.from_numpy(out)[None].to(mask.device), info)
