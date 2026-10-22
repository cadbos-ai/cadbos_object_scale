import torch.nn.functional as F


def _resize_mask(m, h, w):
    if m.shape[0] == h and m.shape[1] == w:
        return m
    return F.interpolate(m[None, None], size=(h, w), mode="bilinear",
                         align_corners=False)[0, 0]
