import torch
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask


class MaskKeepOutside:
    """Оставляет только те области маски, которые заметной частью выходят
    за пределы region. Области, целиком лежащие внутри region, отбрасываются."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "region": ("MASK", {"tooltip": "Оболочка старого объекта (узел 289)."}),
            "min_outside": ("FLOAT", {
                "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Какая доля области должна лежать вне region, чтобы её оставить."}),
            "min_area": ("INT", {"default": 64, "min": 0, "max": 100000}),
        }}

    def run(self, mask, region, min_outside, min_area):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        r = region[0].float().to(m.device)
        if tuple(r.shape) != (H, W):
            r = _resize_mask(r, H, W)
        m_np = (m > 0.5).cpu().numpy()
        r_np = (r > 0.5).cpu().numpy()
        out = np.zeros((H, W), dtype=np.float32)
        lab, n = ndi.label(m_np)
        kept = 0
        for i in range(1, n + 1):
            comp = lab == i
            area = int(comp.sum())
            if area < min_area:
                continue
            if float((comp & ~r_np).sum()) / area >= float(min_outside):
                out[comp] = 1.0
                kept += 1
        info = f"областей: {n}; оставлено (выходят за объект): {kept}"
        return (torch.from_numpy(out)[None].to(mask.device), info)
