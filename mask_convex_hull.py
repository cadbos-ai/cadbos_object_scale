from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
import numpy as np
import torch


class MaskConvexHull:
    """Заполняет выпуклую оболочку маски: объект, которого SAM выделил частично
    (например, только каркас кровати), закрывается целиком."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    def run(self, mask):
        m = mask.float()
        if m.dim() == 2:
            m = m[None]
        out = []
        for i in range(m.shape[0]):
            a = (m[i] > 0.5).cpu().numpy()
            H, W = a.shape
            res = a.astype(np.float32)
            ys, xs = np.nonzero(a)
            if len(xs) >= 3:
                pts = np.stack([xs, ys], axis=1).astype(np.float64)
                try:
                    hull = ConvexHull(pts)
                    poly = [(float(pts[v][0]), float(pts[v][1])) for v in hull.vertices]
                    img = Image.new("L", (W, H), 0)
                    ImageDraw.Draw(img).polygon(poly, fill=255)
                    res = np.maximum(res, np.asarray(img, dtype=np.float32) / 255.0)
                except Exception:
                    pass  # вырожденная маска (линия) — оставить как есть
            out.append(torch.from_numpy(res))
        return (torch.stack(out).to(mask.device),)
