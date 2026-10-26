from .object_scale_placer import ObjectScalePlacer
from .mask_keep_in_front import MaskKeepInFront
from .reference_scale_for_crop import ReferenceScaleForCrop


NODE_CLASS_MAPPINGS = {
    "ObjectScalePlacer": ObjectScalePlacer,
    "MaskKeepInFront": MaskKeepInFront,
    "ReferenceScaleForCrop": ReferenceScaleForCrop,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ObjectScalePlacer": "Object Scale Placer (cadbos)",
    "MaskKeepInFront": "Mask Keep In Front (depth)",
    "ReferenceScaleForCrop": "Reference Scale For Crop",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
