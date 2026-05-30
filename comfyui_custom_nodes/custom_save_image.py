from __future__ import annotations

import os

from PIL import Image


class CustomSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "save_dir": ("STRING", {"default": "outputs/custom"}),
                "file_name": ("STRING", {"default": "sample"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/save"

    def save_images(self, image, save_dir, file_name):
        os.makedirs(save_dir, exist_ok=True)
        batch_size = int(image.shape[0])
        for index in range(batch_size):
            array = (image[index].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(array).save(os.path.join(save_dir, f"{file_name}_{index:02d}.png"))
        return ()


NODE_CLASS_MAPPINGS = {"CustomSaveImage": CustomSaveImage}
NODE_DISPLAY_NAME_MAPPINGS = {"CustomSaveImage": "Custom Save Image (Batch)"}
