class ExcludeWordFromList:
    """Убирает из списка GroundingDINO ('a . b . c') все пункты, содержащие
    любое слово из word (например, 'dining table' убирает все '... table')."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "forceInput": True}),
            "word": ("STRING", {"forceInput": True}),
        }}

    def run(self, text, word):
        tokens = [t for t in str(word).lower().replace(",", " ").split() if len(t) > 2]
        items = [s.strip() for s in str(text).split(".") if s.strip()]
        keep = [s for s in items if not any(t in s.lower() for t in tokens)]
        return (" . ".join(keep),)
