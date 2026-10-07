from .object_scale_placer import ObjectScalePlacer


NODE_CLASS_MAPPINGS = {
  "ObjectScalePlacer": ObjectScalePlacer
}
NODE_DISPLAY_NAME_MAPPINGS = {
  "ObjectScalePlacer": "Object Scale Placer (cadbos)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
