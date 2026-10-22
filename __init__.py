from .object_scale_placer import ObjectScalePlacer
from .mask_keep_in_front import MaskKeepInFront


NODE_CLASS_MAPPINGS = {
    "ObjectScalePlacer": ObjectScalePlacer,
    "MaskKeepInFront": MaskKeepInFront,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ObjectScalePlacer": "Object Scale Placer (cadbos)",
    "MaskKeepInFront": "Mask Keep In Front (depth)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
