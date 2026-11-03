class MaskFallback:
    """Возвращает primary, если в ней что-то есть, иначе fallback."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "primary": ("MASK",),
            "fallback": ("MASK",),
            "min_area": ("INT", {"default": 256, "min": 0, "max": 1000000}),
        }}

    def run(self, primary, fallback, min_area):
        if float((primary > 0.5).sum().item()) >= float(min_area):
            return (primary, "использован основной поиск")
        return (fallback, "основной поиск пуст — использован запасной (ниже порог)")
