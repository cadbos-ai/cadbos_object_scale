class StringFallback:
    """Возвращает primary, если она не пустая, иначе fallback."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "primary": ("STRING", {"forceInput": True}),
            "fallback": ("STRING", {"forceInput": True}),
        }}

    def run(self, primary, fallback):
        p = str(primary).strip()
        if p:
            return (p,)
        f = str(fallback).strip().strip(".").strip()
        return (f,)
