import torch
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask, _dilate, _push_pull_fill


class FillItemsOnObject:
    """Стирает с картинки предметы, лежащие НА объекте (посуда, книги, подушки),
    заполняя их место окружающим цветом объекта. Предметы, стоящие на полу
    перед объектом, не трогает."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "items": ("MASK", {"tooltip": "Найденные мелкие предметы (GroundingDINO + SAM)."}),
            "object_mask": ("MASK", {"tooltip": "Старый объект (узел 61)."}),
            "floor_tolerance": ("FLOAT", {
                "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Предмет считается лежащим на объекте, если его низ выше "
                           "низа объекта больше чем на эту долю высоты объекта."}),
            "grow": ("INT", {"default": 6, "min": 0, "max": 64}),
            "min_area": ("INT", {"default": 32, "min": 0, "max": 100000}),
        },
        "optional": {
            "background": ("IMAGE", {
                "tooltip": "Сцена без старого объекта (узел 220). То, что торчит над "
                           "объектом, заполняется отсюда, а не цветом объекта."}),
        }}

    def run(self, image, items, object_mask, floor_tolerance, grow, min_area,
            background=None):
        img = image[0].float()
        dev = img.device
        H, W, _ = img.shape
        it = items.float().to(dev)
        if it.dim() == 3:
            it = it.amax(dim=0)
        it = (_resize_mask(it, H, W) > 0.5).float()
        obj = (_resize_mask(object_mask[0].float().to(dev), H, W) > 0.5).float()
        hole = torch.zeros((H, W), device=dev)
        info = "предметов на объекте не найдено"
        if obj.sum() > 0 and it.sum() > 0:
            it = it * _dilate(obj, 32)  # только то, что в пределах объекта
            o_np = obj.cpu().numpy() > 0.5
            ys = np.where(o_np.any(axis=1))[0]
            floor_line = float(ys[-1]) - float(floor_tolerance) * float(ys[-1] - ys[0] + 1)
            lab, n = ndi.label(it.cpu().numpy() > 0.5)
            keep = np.zeros((H, W), dtype=np.float32)
            k = 0
            for i in range(1, n + 1):
                comp = lab == i
                if int(comp.sum()) < min_area:
                    continue
                if float(np.where(comp.any(axis=1))[0][-1]) < floor_line:
                    keep[comp] = 1.0
                    k += 1
            if k > 0:
                hole = _dilate(torch.from_numpy(keep).to(dev), int(grow))
                # внутри контура объекта — заливка только цветом самого объекта
                img = _push_pull_fill(img, hole, exclude=(1 - obj))
                if background is not None:
                    # выше верхнего края объекта (в каждом столбце) — фон без объекта
                    top = np.where(o_np.any(axis=0), o_np.argmax(axis=0), H)
                    above = torch.from_numpy(
                        (np.arange(H)[:, None] < top[None, :]).astype(np.float32)).to(dev)
                    bg = background[0].float().to(dev)[..., :3]
                    if bg.shape[0] != H or bg.shape[1] != W:
                        bg = F.interpolate(bg.permute(2, 0, 1)[None], size=(H, W),
                                           mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
                    out = (hole * above)[..., None]
                    img = img * (1 - out) + bg * out
            info = f"найдено областей: {n}; стёрто как лежащие на объекте: {k}"
        return (img[None].clamp(0, 1), hole[None], info)
