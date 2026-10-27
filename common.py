import math
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- helpers

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


def _dilate(m, r):
    if r <= 0:
        return m
    return F.max_pool2d(m[None, None], 2 * r + 1, 1, r)[0, 0]


def _push_pull_fill(img, hole, exclude=None):
    """Заливка области hole (H,W; 1 = заполнить) цветами окружения.
    img: (H,W,C) 0..1. Пирамидальное усреднение (push-pull): быстро,
    без дополнительных библиотек, даёт мягкий фон без следов объекта."""
    known = (1.0 - hole).clamp(0, 1)
    if exclude is not None:          # эти пиксели не используются как образец фона
        known = known * (1.0 - exclude)
    if known.sum() < 1:
        return img
    x = img.permute(2, 0, 1)[None] * known[None, None]
    w = known[None, None]
    xs, ws = [x], [w]
    while min(xs[-1].shape[-2:]) > 1:
        x, w = xs[-1], ws[-1]
        h_, w_ = x.shape[-2:]
        ph, pw = h_ % 2, w_ % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
            w = F.pad(w, (0, pw, 0, ph))
        xs.append(F.avg_pool2d(x, 2))
        ws.append(F.avg_pool2d(w, 2))
    col = xs[-1] / ws[-1].clamp_min(1e-8)
    for lvl in range(len(xs) - 2, -1, -1):
        x, w = xs[lvl], ws[lvl]
        up = F.interpolate(col, size=x.shape[-2:], mode="bilinear",
                           align_corners=False)
        cur = x / w.clamp_min(1e-8)
        a = w.clamp(0, 1)
        col = cur * a + up * (1 - a)
    col = col[0].permute(1, 2, 0)
    h3 = hole[..., None]
    return img * (1 - h3) + col * h3


def _erode(m, r):
    if r <= 0:
        return m
    return 1.0 - _dilate(1.0 - m, r)


def _blur(m, r):
    if r <= 0:
        return m
    x = F.pad(m[None, None], (r, r, r, r), mode="replicate")
    return F.avg_pool2d(x, 2 * r + 1, 1)[0, 0]


def _auto_reference_mask(ref, tol):
    """Маска объекта на референсе с однотонным фоном (каталожное фото).
    Фон = пиксели, близкие к цвету рамки кадра и связанные с краем кадра.
    Внутренние светлые детали объекта (белые подушки и т.п.) остаются
    объектом, т.к. не связаны с краем."""
    H, W, _ = ref.shape
    border = torch.cat([ref[0], ref[-1], ref[:, 0], ref[:, -1]], 0)
    bg_color = border.median(dim=0).values
    dist = (ref - bg_color).abs().max(dim=-1).values
    cand = (dist < tol).float()                      # похоже на фон

    # поиск фона, связанного с краем, на уменьшенной копии (быстро)
    s = min(1.0, 384.0 / max(H, W))
    h, w = max(8, int(round(H * s))), max(8, int(round(W * s)))
    cs = (F.interpolate(cand[None, None], size=(h, w), mode="area") > 0.5).float()
    reach = torch.zeros_like(cs)
    reach[..., 0, :] = cs[..., 0, :]
    reach[..., -1, :] = cs[..., -1, :]
    reach[..., :, 0] = cs[..., :, 0]
    reach[..., :, -1] = cs[..., :, -1]
    for _ in range(4 * (h + w)):
        new = F.max_pool2d(reach, 3, 1, 1) * cs
        if torch.equal(new, reach):
            break
        reach = new
    reach_full = F.interpolate(reach, size=(H, W), mode="nearest")[0, 0]
    reach_full = _dilate(reach_full, max(1, int(math.ceil(1.0 / s))))
    bg = reach_full * cand
    obj = 1.0 - bg
    # убрать мелкий мусор (пылинки, артефакты JPEG)
    obj = _dilate(_erode(obj, 2), 2)
    return obj


def _draw_rect(img, x0, y0, x1, y1, t, color):
    c = torch.tensor(color, dtype=img.dtype, device=img.device)
    H, W, _ = img.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    t = max(1, min(t, (x1 - x0) // 2, (y1 - y0) // 2))
    img[y0:y0 + t, x0:x1] = c
    img[y1 - t:y1, x0:x1] = c
    img[y0:y1, x0:x0 + t] = c
    img[y0:y1, x1 - t:x1] = c
    return img
