import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask


class MaskRecoverThin:
    """Возвращает в маску узкие детали объекта, которые пропустил SAM.
    Ищет их по отличию кадра с объектом от чистого фона."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK", {"tooltip": "Маска объекта (узел 252)."}),
            "image": ("IMAGE", {"tooltip": "Кадр с новым объектом (узел 101)."}),
            "clean": ("IMAGE", {"tooltip": "Тот же кадр без объекта (узел 220)."}),
            "threshold": ("FLOAT", {
                "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
                "tooltip": "Насколько пиксель должен отличаться от чистого фона."}),
            "reach": ("INT", {
                "default": 48, "min": 1, "max": 256,
                "tooltip": "Как далеко от маски искать детали, px."}),
            "max_width": ("INT", {
                "default": 4, "min": 1, "max": 32,
                "tooltip": "Более широкие области считаются изменением фона, а не деталью. "
                           "Предельная ширина детали примерно равна 2 x значение + 1, px."}),
            "min_area": ("INT", {"default": 8, "min": 0, "max": 10000}),
        }}

    def run(self, mask, image, clean, threshold, reach, max_width, min_area):
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
        if not m.any():
            return (mask, "маска пустая — ничего не добавлено")
        img = img_t.cpu().numpy()
        cln = cln_t.cpu().numpy()

        diff = np.abs(img - cln).max(axis=-1)
        wts = np.array([0.299, 0.587, 0.114])
        l_i = (img * wts).sum(-1) + 1e-3
        l_c = (cln * wts).sum(-1) + 1e-3
        chroma = np.abs(img / l_i[..., None] - cln / l_c[..., None]).max(axis=-1)
        ratio = l_i / l_c
        # тень: темнее фона, но с тем же оттенком и не почти чёрная
        shadow = (ratio < 1.0) & (ratio > 0.35) & (chroma < 0.15)

        near = ndi.binary_dilation(m, iterations=int(reach))
        cand = (diff > float(threshold)) & ~shadow & near & ~m

        # широкие области (изменённый пол) отбрасываем, оставляем узкие детали
        wide = ndi.binary_opening(cand, iterations=int(max_width))
        thin = cand & ~ndi.binary_dilation(wide, iterations=1)

        # только то, что соединено с объектом
        st = np.ones((3, 3))
        lab, _ = ndi.label(m | thin, structure=st)
        ids = np.unique(lab[m])
        thin &= np.isin(lab, ids[ids > 0])

        lab2, n = ndi.label(thin, structure=st)
        kept = np.zeros_like(thin)
        count = 0
        for i in range(1, n + 1):
            comp = lab2 == i
            if int(comp.sum()) >= int(min_area):
                kept |= comp
                count += 1

        out = (m | kept).astype(np.float32)
        info = f"добавлено областей: {count}; пикселей: {int(kept.sum())}"
        return (torch.from_numpy(out)[None].to(mask.device), info)
