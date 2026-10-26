import torch
import torch.nn.functional as F


def _resize_mask(m, h, w):
    if m.shape[0] == h and m.shape[1] == w:
        return m
    return F.interpolate(m[None, None], size=(h, w), mode="bilinear",
                         align_corners=False)[0, 0]

def _bbox(m, thr=0.5):
    """Габаритная рамка маски (H,W) -> (x0, y0, x1, y1) или None."""
    ys, xs = torch.nonzero(m > thr, as_tuple=True)
    if ys.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
