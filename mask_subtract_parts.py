import torch
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask


class MaskSubtractParts:
    """Вычитает из base только небольшие области parts (например, стулья перед
    столом). Область, накрывающая заметную часть base, не вычитается; всего
    вычитается не больше max_total от base. Выход removed — что было вычтено."""
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
            "max_total": ("FLOAT", {
                "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Всего можно вычесть не больше этой доли base."}),
        }}

    def run(self, base, parts, max_fraction, max_total=0.35):
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
        cut, skipped = 0, 0
        if total > 0:
            lab, n = ndi.label(p_np)
            comps = []
            for i in range(1, n + 1):
                comp = lab == i
                ov = float((comp & b_np).sum())
                if ov > 0:
                    comps.append((ov, comp))
            comps.sort(key=lambda c: c[0])  # сначала самые маленькие наложения
            used = 0.0
            for ov, comp in comps:
                if ov / total > float(max_fraction) or (used + ov) / total > float(max_total):
                    skipped += 1  # сам объект под другим названием или его крупная часть
                    continue
                out &= ~comp
                removed |= comp
                used += ov
                cut += 1
        info = (f"вычтено областей: {cut}; пропущено: {skipped}; "
                f"вычтено от объекта: {0 if total == 0 else 100 * float((b_np & ~out).sum()) / total:.0f}%")
        to_t = lambda a: torch.from_numpy(a.astype(np.float32))[None].to(base.device)
        return (to_t(out), info, to_t(removed))
