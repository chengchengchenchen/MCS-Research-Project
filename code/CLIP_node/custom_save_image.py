import os
from PIL import Image
import datetime
import torch

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

        batch_size = image.shape[0]
        for i in range(batch_size):
            img_np = (image[i].cpu().numpy() * 255).astype("uint8")
            img = Image.fromarray(img_np)

            save_path = os.path.join(save_dir, f"{file_name}_{i:02d}.png")
            img.save(save_path)

        print(f"[CustomSaveImage] Finished saving {batch_size} images to '{save_dir}'")
        return ()

NODE_CLASS_MAPPINGS = {
    "CustomSaveImage": CustomSaveImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomSaveImage": "Custom Save Image (Batch)"
}
