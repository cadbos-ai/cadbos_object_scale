from .object_scale_placer import ObjectScalePlacer
from .mask_keep_in_front import MaskKeepInFront
from .reference_scale_for_crop import ReferenceScaleForCrop
from .fill_items_on_object import FillItemsOnObject


NODE_CLASS_MAPPINGS = {
    "ObjectScalePlacer": ObjectScalePlacer,
    "MaskKeepInFront": MaskKeepInFront,
    "ReferenceScaleForCrop": ReferenceScaleForCrop,
    "FillItemsOnObject": FillItemsOnObject,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ObjectScalePlacer": "Object Scale Placer (cadbos)",
    "MaskKeepInFront": "Mask Keep In Front (depth)",
    "ReferenceScaleForCrop": "Reference Scale For Crop",
    "FillItemsOnObject": "Fill Items On Object",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
