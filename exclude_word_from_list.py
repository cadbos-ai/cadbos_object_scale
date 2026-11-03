class ExcludeWordFromList:
    """Убирает из списка GroundingDINO ('a . b . c') все пункты, содержащие
    любое слово из word или его близкие синонимы (стол — desk и т.п.)."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    SYN = {
        "table": ["desk", "counter", "bench"],
        "desk": ["table"],
        "chair": ["stool", "seat"],
        "armchair": ["chair", "sofa"],
        "sofa": ["couch", "settee", "bench", "armchair"],
        "couch": ["sofa", "settee"],
        "bed": ["sofa", "mattress"],
        "cabinet": ["wardrobe", "dresser", "cupboard", "sideboard", "shelf"],
        "wardrobe": ["cabinet", "dresser", "cupboard"],
        "dresser": ["cabinet", "nightstand", "chest"],
        "nightstand": ["dresser", "cabinet", "side table"],
        "shelf": ["bookshelf", "cabinet"],
        "bookshelf": ["shelf", "cabinet"],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "forceInput": True}),
            "word": ("STRING", {"forceInput": True}),
        }}

    def run(self, text, word):
        tokens = [t for t in str(word).lower().replace(",", " ").split() if len(t) > 2]
        for t in list(tokens):
            tokens += self.SYN.get(t, [])
        items = [s.strip() for s in str(text).split(".") if s.strip()]
        keep = [s for s in items if not any(t in s.lower() for t in tokens)]
        return (" . ".join(keep),)
