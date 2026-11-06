from .common import _resize_mask, _bbox


class MaskClearTop:
    """Убирает маску защиты выше заданного уровня рамки объекта.
    Нужно, когда верх объекта (столешница) лежит поверх защищённых предметов."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK", {"tooltip": "Маска защиты (узел 314)."}),
            "region": ("MASK", {"tooltip": "Старый объект (узел 61)."}),
            "top_fraction": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Какая верхняя доля высоты рамки объекта не защищается. "
                           "0 — маска не меняется."}),
        }}

    def run(self, mask, region, top_fraction):
        m = mask.float().clone()
        if m.dim() == 2:
            m = m[None]
        H, W = m.shape[-2:]
        r = _resize_mask(region[0].float().to(m.device), H, W)
        bb = _bbox(r)
        if bb is None or top_fraction <= 0:
            return (m, "маска не изменена")
        y0, y1 = bb[1], bb[3]
        cut = int(round(y0 + (y1 - y0) * float(top_fraction)))
        m[:, :cut, :] = 0
        return (m, f"защита снята выше строки {cut} (рамка объекта: {y0}–{y1})")
