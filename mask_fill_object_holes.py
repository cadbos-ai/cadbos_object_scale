import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask


class MaskFillObjectHoles:
    """Закрывает отверстия в маске, если там на кадре виден объект,
    и оставляет настоящие просветы, где виден чистый фон."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK", {"tooltip": "Маска объекта (узел 341)."}),
            "image": ("IMAGE", {"tooltip": "Кадр с новым объектом (узел 101)."}),
            "clean": ("IMAGE", {"tooltip": "Тот же кадр без объекта (узел 220)."}),
            "threshold": ("FLOAT", {
                "default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01,
                "tooltip": "Насколько пиксель должен отличаться от чистого фона."}),
            "min_changed": ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Какая доля отверстия должна отличаться от фона, чтобы его закрыть."}),
            "always_fill_below": ("INT", {
                "default": 64, "min": 0, "max": 100000,
                "tooltip": "Отверстия меньше этой площади закрываются всегда, px."}),
        }}

    def run(self, mask, image, clean, threshold, min_changed, always_fill_below):
        m_t = mask.float()
        if m_t.dim() == 3:
            m_t = m_t.amax(dim=0)
        H, W = m_t.shape
        img_t = image[0].float()[..., :3].to(m_t.device)
        cln_t = clean[0].float()[..., :3].to(m_t.device)
        if img_t.shape[0] != H or img_t.shape[1] != W:
            img_t = F.interpolate(img_t.permute(2, 0, 1)[None], size=(H, W), mode="bilinear",
                                  align_corners=False)[0].permute(1, 2, 0)
        if cln_t.shape[0] != H or cln_t.shape[1] != W:
            cln_t = F.interpolate(cln_t.permute(2, 0, 1)[None], size=(H, W), mode="bilinear",
                                  align_corners=False)[0].permute(1, 2, 0)

        m = (_resize_mask(m_t, H, W) > 0.5).cpu().numpy()
        diff = np.abs(img_t.cpu().numpy() - cln_t.cpu().numpy()).max(axis=-1)
        holes = ndi.binary_fill_holes(m) & ~m
        lab, n = ndi.label(holes)
        out = m.copy()
        filled = 0
        for i in range(1, n + 1):
            comp = lab == i
            changed = float((diff[comp] > float(threshold)).mean())
            if int(comp.sum()) < int(always_fill_below) or changed >= float(min_changed):
                out |= comp
                filled += 1
        info = f"отверстий: {n}; закрыто: {filled}; оставлено как просвет: {n - filled}"
        return (torch.from_numpy(out.astype(np.float32))[None].to(mask.device), info)
