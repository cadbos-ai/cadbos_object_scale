import torch
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask


class MaskSubtractParts:
    """Вычитает из base только небольшие области parts. Область, которая
    накрывает заметную часть base (тот же объект, найденный под другим
    названием), не вычитается. Выход removed — то, что было вычтено."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING", "MASK")
    RETURN_NAMES = ("mask", "info", "removed")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("MASK",),
            "parts": ("MASK",),
            "max_fraction": ("FLOAT", {
                "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Область parts не вычитается, если накрывает больше этой "
                           "доли base."}),
        }}

    def run(self, base, parts, max_fraction):
        b = base.float()
        if b.dim() == 3:
            b = b.amax(dim=0)
        H, W = b.shape
        p = parts.float().to(b.device)
        if p.dim() == 3:
            p = p.amax(dim=0)
        if tuple(p.shape) != (H, W):
            p = _resize_mask(p, H, W)
        b_np = b.cpu().numpy() > 0.5
        p_np = p.cpu().numpy() > 0.5
        total = float(b_np.sum())
        out = b_np.copy()
        removed = np.zeros((H, W), dtype=bool)
        lab, n = ndi.label(p_np)
        cut, skipped = 0, 0
        for i in range(1, n + 1):
            comp = lab == i
            ov = float((comp & b_np).sum())
            if ov == 0:
                continue
            if total > 0 and ov / total > float(max_fraction):
                skipped += 1  # это сам объект под другим названием
                continue
            out &= ~comp
            removed |= comp
            cut += 1
        info = f"вычтено областей: {cut}; пропущено (сам объект): {skipped}"
        to_t = lambda a: torch.from_numpy(a.astype(np.float32))[None].to(base.device)
        return (to_t(out), info, to_t(removed))
