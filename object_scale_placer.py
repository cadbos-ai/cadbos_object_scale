"""
cadbos_object_scale — узел ComfyUI для управления масштабом вставляемого объекта
в схеме obj-replace (Qwen-Image-Edit-2511).

Узел "Object Scale Placer (cadbos)":
  * берёт сцену, маску старого объекта и референс нового объекта;
  * вычисляет, какого размера и где должен стоять новый объект
    (scale_percent = 0 → автоподгонка под рамку старого объекта,
     scale_percent > 0 → процент от автоподгонки);
  * стирает старый объект (заливка фоном), вклеивает вырезку референса
    нужного размера ("collage") или рисует зелёную рамку ("frame");
  * отдаёт маску области перерисовки, охватывающую и старый, и новый объект.

Установка: положить папку cadbos_object_scale в ComfyUI/custom_nodes/
и перезапустить ComfyUI. Дополнительные пакеты не нужны (только torch).
"""

import math
import torch
import torch.nn.functional as F

from .common import _resize_mask


# ----------------------------------------------------------------- helpers

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


def _erode(m, r):
    if r <= 0:
        return m
    return 1.0 - _dilate(1.0 - m, r)


def _blur(m, r):
    if r <= 0:
        return m
    x = F.pad(m[None, None], (r, r, r, r), mode="replicate")
    return F.avg_pool2d(x, 2 * r + 1, 1)[0, 0]


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


# -------------------------------------------------------------------- node

class ObjectScalePlacer:
    CATEGORY = "cadbos/object"
    FUNCTION = "place"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("guide_image", "union_mask", "object_mask", "preview", "info", "protect")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("IMAGE",),
                "target_mask": ("MASK",),
                "reference": ("IMAGE",),
                "scale_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 400.0, "step": 1.0,
                    "tooltip": "0 = автоматически (вписать в рамку старого объекта). "
                               "100 = как авто, 150 = в 1,5 раза больше, 70 = на 30% меньше."}),
                "fit": (["contain", "height", "width"], {
                    "default": "contain",
                    "tooltip": "Как считать авторазмер: contain — целиком внутри рамки "
                               "старого объекта; height — по высоте; width — по ширине."}),
                "anchor": (["bottom", "center", "top"], {
                    "default": "bottom",
                    "tooltip": "Какая точка старого объекта остаётся на месте: "
                               "bottom — опора на пол, center — центр, top — подвес к потолку."}),
                "mode": (["collage", "frame"], {
                    "default": "collage",
                    "tooltip": "collage — вклеить вырезку референса; "
                               "frame — нарисовать зелёную рамку нужного размера."}),
                "erase_grow": ("INT", {
                    "default": 6, "min": 0, "max": 64,
                    "tooltip": "На сколько пикселей расширить зону стирания старого объекта."}),
                "frame_thickness": ("INT", {"default": 14, "min": 2, "max": 64}),
                "bg_tolerance": ("FLOAT", {
                    "default": 0.08, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "Чувствительность автоотделения фона референса "
                               "(если reference_mask не подключена)."}),
            },
            "optional": {
                "reference_mask": ("MASK", {
                    "tooltip": "Маска объекта на референсе (белое = объект), например из RMBG."}),
                "protect_mask": ("MASK", {
                    "tooltip": "Что нельзя закрывать (предметы на переднем плане). "
                               "Подключите выход узла 63."}),
                "background": ("IMAGE", {
                    "tooltip": "Сцена с уже удалённым старым объектом (узел 220). "
                               "Если подключено — используется вместо размытой заливки."}),
                "depth": ("IMAGE", {
                    "tooltip": "Карта глубины сцены (узел 105)."}),
                "occluders": ("MASK", {
                    "tooltip": "Найденные предметы сцены без старого объекта."}),
                "depth_margin": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Насколько предмет должен быть ближе переднего края "
                               "объекта, чтобы считаться передним планом."}),
                "depth_near_is_bright": ("BOOLEAN", {"default": True}),
                "h_anchor": (["center", "left", "right"], {
                    "default": "center",
                    "tooltip": "Какой край по горизонтали остаётся на месте: "
                               "left — левый (объект растёт вправо), right — правый, "
                               "center — центр."}),
                "offset_x": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 1,
                    "tooltip": "Сдвиг по горизонтали, px сцены (+ вправо)."}),
                "offset_y": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 1,
                    "tooltip": "Сдвиг по вертикали, px сцены (+ вниз)."}),
                "shadow_source": ("IMAGE", {
                    "tooltip": "Чистый фон, соответствующий reference (узел 220). "
                               "Тень объекта переносится с reference и масштабируется вместе с ним."}),
                "shadow_margin": ("INT", {
                    "default": 48, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Насколько далеко от объекта искать его тень, px."}),
                "shadow_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Сила перенесённой тени: 0 — нет, 1 — как на этапе 1."}),
                "shadow_reach": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Переносить только тень у самого объекта (контактную), px. "
                               "0 — переносить всю тень."}),
            },
        }

    def place(self, scene, target_mask, reference, scale_percent, fit, anchor,
              mode, erase_grow, frame_thickness, bg_tolerance,
              reference_mask=None, protect_mask=None, background=None,
              depth=None, occluders=None, depth_margin=0.03,
              depth_near_is_bright=True, h_anchor="center",
              offset_x=0, offset_y=0, shadow_source=None,
              shadow_margin=48, shadow_strength=1.0, shadow_reach=0):
        img = scene[0].clone().float()
        H, W, _ = img.shape
        dev = img.device

        old = _resize_mask(target_mask[0].float().to(dev), H, W)
        old = (old > 0.5).float()
        bb_old = _bbox(old)
        if bb_old is None:
            raise ValueError("Object Scale Placer: маска старого объекта пустая — "
                             "GroundingDINO не нашёл объект. Проверьте название объекта.")
        ox0, oy0, ox1, oy1 = bb_old
        ow, oh = ox1 - ox0, oy1 - oy0

        prot = torch.zeros((H, W), device=dev)
        if protect_mask is not None:
            prot = (_resize_mask(protect_mask[0].float().to(dev), H, W) > 0.5).float()

        # --- вырезка референса
        ref = reference[0].float().to(dev)
        RH, RW, _ = ref.shape
        if reference_mask is not None:
            rm = _resize_mask(reference_mask[0].float().to(dev), RH, RW).clamp(0, 1)
            src = "reference_mask"
        else:
            rm = _auto_reference_mask(ref, bg_tolerance)
            src = "auto (однотонный фон)"
        bb_ref = _bbox(rm)
        if bb_ref is None:
            raise ValueError("Object Scale Placer: на референсе не найден объект. "
                             "Подключите reference_mask (RMBG) или измените bg_tolerance.")
        rx0, ry0, rx1, ry1 = bb_ref
        ref_c = ref[ry0:ry1, rx0:rx1]
        rm_c = rm[ry0:ry1, rx0:rx1]
        rw, rh = rx1 - rx0, ry1 - ry0

        # --- размер
        if fit == "height":
            s_auto = oh / rh
        elif fit == "width":
            s_auto = ow / rw
        else:
            s_auto = min(ow / rw, oh / rh)
        factor = 1.0 if scale_percent <= 0 else scale_percent / 100.0
        s = s_auto * factor
        nw, nh = rw * s, rh * s
        k = min(1.0, W / nw, H / nh)  # не больше кадра
        nw, nh = max(2, int(round(nw * k))), max(2, int(round(nh * k)))

        # --- положение
        if h_anchor == "left":
            nx0 = ox0
        elif h_anchor == "right":
            nx0 = ox1 - nw
        else:
            nx0 = int(round((ox0 + ox1) / 2.0 - nw / 2.0))
        if anchor == "bottom":
            ny0 = oy1 - nh
        elif anchor == "top":
            ny0 = oy0
        else:
            ny0 = int(round((oy0 + oy1) / 2.0 - nh / 2.0))
        nx0 += int(offset_x)
        ny0 += int(offset_y)
        nx0 = max(0, min(W - nw, nx0))
        ny0 = max(0, min(H - nh, ny0))
        nx1, ny1 = nx0 + nw, ny0 + nh

        # --- 0) доп. защита: предметы ПЕРЕД новым объектом (по глубине)
        extra_px = 0
        if occluders is not None and depth is not None:
            occ = occluders.float().to(dev)
            if occ.dim() == 3:
                occ = occ.amax(dim=0)
            occ = (_resize_mask(occ, H, W) > 0.5).float()
            dep = depth[0].float().to(dev)
            if dep.dim() == 3:
                dep = dep.mean(dim=-1)
            dep = _resize_mask(dep, H, W)
            if not depth_near_is_bright:
                dep = 1.0 - dep
            vals = dep[old > 0.5]
            if vals.numel() > 0:
                d_front = torch.quantile(vals, 0.9)  # передний край объекта
                zone = torch.zeros((H, W), device=dev)
                zone[ny0:ny1, nx0:nx1] = 1.0
                zone = _dilate(zone, 8)
                extra = occ * zone * (dep > d_front + depth_margin).float() * (1 - old)
                extra = _dilate(_erode(extra, 2), 6)  # убрать крошки, чуть расширить
                extra_px = int(extra.sum().item())
                prot = torch.maximum(prot, extra)

        # --- 1) фон без старого объекта
        if background is not None:
            bgi = background[0].float().to(dev)[..., :3]
            if bgi.shape[0] != H or bgi.shape[1] != W:
                bgi = F.interpolate(bgi.permute(2, 0, 1)[None], size=(H, W),
                                    mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            guide = bgi.clone()
            bg_src = "чистый (background)"
        else:
            hole = _dilate(old, erase_grow) * (1 - prot)
            guide = _push_pull_fill(img, hole, exclude=_dilate(prot, 4))
            bg_src = "размытая заливка"

        # --- 2) поставить новый объект
        box = torch.zeros((H, W), device=dev)
        box[ny0:ny1, nx0:nx1] = 1.0
        obj_mask = torch.zeros((H, W), device=dev)
        shadow_px = 0
        if mode == "collage" and shadow_source is not None and shadow_strength > 0:
            # --- перенос тени: затемнение reference относительно чистого фона,
            #     масштабированное и сдвинутое вместе с объектом
            sh = shadow_source[0].float().to(dev)[..., :3]
            if sh.shape[0] != RH or sh.shape[1] != RW:
                sh = F.interpolate(sh.permute(2, 0, 1)[None], size=(RH, RW), mode="bilinear",
                                   align_corners=False)[0].permute(1, 2, 0)
            m = int(shadow_margin)
            sx0, sy0 = max(0, rx0 - m), max(0, ry0 - m)
            sx1, sy1 = min(RW, rx1 + m), min(RH, ry1 + m)
            wts = torch.tensor([0.299, 0.587, 0.114], device=dev)
            l_ref = (ref[sy0:sy1, sx0:sx1, :3] * wts).sum(-1)
            l_bg = (sh[sy0:sy1, sx0:sx1] * wts).sum(-1)
            ratio = (l_ref + 1e-3) / (l_bg + 1e-3)
            objm = _dilate((rm[sy0:sy1, sx0:sx1] > 0.5).float(), 2)
            f = max(2, m // 3)
            inner = torch.zeros_like(ratio)
            if ratio.shape[0] > 2 * f and ratio.shape[1] > 2 * f:
                inner[f:-f, f:-f] = 1.0
            border = (1 - inner) * (1 - objm)
            if border.sum() > 16:  # выравнивание общей яркости этапа 1 и чистого фона
                ratio = ratio / torch.quantile(ratio[border > 0.5], 0.75).clamp_min(1e-3)
            ratio = ratio.clamp(0, 1)
            ratio = _push_pull_fill(ratio[..., None], objm)[..., 0]  # под объектом — по соседям
            ratio = _blur(ratio, 4)
            dark = ((1 - ratio) - 0.05).clamp_min(0) / 0.95
            if shadow_reach > 0:  # оставить только контактную тень у объекта
                near = _dilate((rm[sy0:sy1, sx0:sx1] > 0.5).float(), int(shadow_reach))
                dark = dark * _blur(near, max(2, int(shadow_reach) // 2))
            fx, fy = nw / rw, nh / rh
            dh = max(2, int(round((sy1 - sy0) * fy)))
            dw = max(2, int(round((sx1 - sx0) * fx)))
            dark_s = F.interpolate(dark[None, None], size=(dh, dw), mode="bilinear",
                                   align_corners=False)[0, 0].clamp(0, 1)
            tx0 = nx0 - int(round((rx0 - sx0) * fx))
            ty0 = ny0 - int(round((ry0 - sy0) * fy))
            cx0, cy0 = max(0, tx0), max(0, ty0)
            cx1, cy1 = min(W, tx0 + dw), min(H, ty0 + dh)
            if cx1 > cx0 and cy1 > cy0:
                d = dark_s[cy0 - ty0:cy1 - ty0, cx0 - tx0:cx1 - tx0]
                d = d * (1 - prot[cy0:cy1, cx0:cx1])
                guide[cy0:cy1, cx0:cx1] = guide[cy0:cy1, cx0:cx1] * (1 - d[..., None])
                shadow_px = int((d > 0.05).sum().item())
        if mode == "collage":
            antialias = nw < rw or nh < rh
            rc = F.interpolate(ref_c.permute(2, 0, 1)[None], size=(nh, nw),
                               mode="bicubic", align_corners=False,
                               antialias=antialias)[0].permute(1, 2, 0).clamp(0, 1)
            am = F.interpolate(rm_c[None, None], size=(nh, nw), mode="bilinear",
                               align_corners=False, antialias=antialias)[0, 0]
            am = _blur(_erode((am > 0.5).float(), 1), 1)  # мягкий край
            am = am * (1 - prot[ny0:ny1, nx0:nx1])
            region = guide[ny0:ny1, nx0:nx1]
            guide[ny0:ny1, nx0:nx1] = region * (1 - am[..., None]) + rc * am[..., None]
            obj_mask[ny0:ny1, nx0:nx1] = am
        else:
            base = guide.clone()
            guide = _draw_rect(guide, nx0, ny0, nx1, ny1, frame_thickness,
                               (0.0, 1.0, 0.0))
            # рамку не рисуем поверх защищённых предметов
            guide = torch.where(prot[..., None] > 0.5, base, guide)
            obj_mask = box.clone()

        # --- 3) маска перерисовки: старый объект + место нового
        union = torch.maximum(_dilate(old, erase_grow), box)

        # --- превью: красная рамка = старый объект, синяя = новый
        prev = guide.clone()
        prev = _draw_rect(prev, ox0, oy0, ox1, oy1, 3, (1.0, 0.0, 0.0))
        prev = _draw_rect(prev, nx0, ny0, nx1, ny1, 3, (0.0, 0.4, 1.0))

        info = (f"режим: {mode}; масштаб: "
                f"{'авто' if scale_percent <= 0 else f'{scale_percent:.0f}% от авто'} "
                f"(fit={fit}, anchor={anchor}, h_anchor={h_anchor}, "
                f"offset={offset_x},{offset_y})\n"
                f"старый объект: {ow}x{oh}px @({ox0},{oy0}); "
                f"новый объект: {nw}x{nh}px @({nx0},{ny0}); "
                f"итоговый коэффициент к референсу: {s * k:.3f}\n"
                f"маска референса: {src}; фон: {bg_src}; "
                f"доп. защита (предметы спереди): {extra_px} px; "
                f"тень перенесена: {shadow_px} px"
                + ("; ВНИМАНИЕ: объект ограничен размером кадра" if k < 1.0 else ""))
        return (guide[None].clamp(0, 1), union[None], obj_mask[None],
                prev[None].clamp(0, 1), info, prot[None])
