import math
import numpy as np

from .common import _resize_mask


def _orientation(m, d, vp_y=0.5):
    """m — bool маска (H,W), d — глубина (H,W, ближе = больше) или None."""
    H, W = m.shape
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None, "маска пустая"
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    ev, evec = np.linalg.eigh(cov)
    l2, l1 = ev[0], ev[1]
    v = evec[:, 1]
    if v[1] > 0:          # ось направлена «вверх» по картинке (вглубь)
        v = -v
    elong = math.sqrt(max(l1, 1e-6) / max(l2, 1e-6))
    ang = math.degrees(math.atan2(-v[1], abs(v[0]) + 1e-9))   # 0 = горизонт, 90 = вертикаль
    vp = np.array([W / 2.0, H * vp_y])
    to_vp = vp - c
    n = np.linalg.norm(to_vp)
    ang_vp = 180.0 if n < 1 else math.degrees(math.acos(
        max(-1.0, min(1.0, float(np.dot(v, to_vp) / n)))))
    spread = 0.0
    if d is not None:
        dv = d[m]
        full = float(np.quantile(d, 0.98) - np.quantile(d, 0.02)) + 1e-6
        spread = float(np.quantile(dv, 0.95) - np.quantile(dv, 0.05)) / full
    info = (f"вытянутость {elong:.2f}; наклон оси {ang:.0f}°; угол к точке схода {ang_vp:.0f}°; "
            f"разброс глубины {spread:.2f}")
    if ang >= 65 or elong < 1.3:      # компактный или вытянутый вертикально: решает глубина
        if spread >= 0.2:
            return "it is seen from its short end, and its long side runs away from the camera toward the back of the room", info
        return "it has no clearly longer side", info
    if ang < 20:
        return "it is seen from its long side, and its long side runs across the image from left to right", info
    if ang_vp <= 30:
        return "it is seen from its short end, and its long side runs away from the camera toward the back of the room", info
    if v[0] > 0:
        return "it stands diagonally, and its long side runs from the front left toward the back right", info
    return "it stands diagonally, and its long side runs from the front right toward the back left", info


class ObjectOrientationText:
    """Определяет по маске старого объекта и карте глубины, как он развёрнут,
    и выдаёт фразу для промпта (замена ответа Qwen-VL)."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK", {"tooltip": "Старый объект (узел 322)."}),
            "depth": ("IMAGE", {"tooltip": "Карта глубины (узел 105)."}),
            "depth_near_is_bright": ("BOOLEAN", {"default": True}),
            "horizon": ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Высота точки схода в долях кадра (0.5 — середина)."}),
        }}

    def run(self, mask, depth, depth_near_is_bright, horizon):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        dep = depth[0].float()
        if dep.dim() == 3:
            dep = dep.mean(dim=-1)
        dep = dep.to(m.device)
        if tuple(dep.shape) != (H, W):
            dep = _resize_mask(dep, H, W)
        if not depth_near_is_bright:
            dep = 1.0 - dep
        phrase, info = _orientation(m.cpu().numpy() > 0.5, dep.cpu().numpy(), float(horizon))
        if phrase is None:
            return ("", info)
        return (f"Required orientation of the new object: {phrase}. "
                f"This overrides the viewing angle of Picture 2.", info)
