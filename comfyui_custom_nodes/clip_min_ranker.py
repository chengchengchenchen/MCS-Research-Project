from __future__ import annotations

import csv
import json
import os

from .calculate_score import CalculateScore


class CLIPMinRanker(CalculateScore):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "candidate_images": ("IMAGE",),
                "save_dir": ("STRING", {"default": "outputs/clip_scores"}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 128}),
                "fusion_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dino_model_name": (
                    [
                        "vit_base_patch14_dinov2.lvd142m",
                        "vit_small_patch14_dinov2.lvd142m",
                        "vit_large_patch14_dinov2.lvd142m",
                    ],
                    {"default": "vit_base_patch14_dinov2.lvd142m"},
                ),
                "dino_pool": (["cls", "mean", "gap", "auto"], {"default": "auto"}),
            },
        }

    def run(
        self,
        source_image,
        candidate_images,
        save_dir: str,
        batch_size: int = 8,
        fusion_weight: float = 0.0,
        dino_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        dino_pool: str = "auto",
    ):
        super().run(source_image, candidate_images, save_dir, batch_size, dino_model_name, dino_pool)
        json_path = os.path.join(save_dir, "scores.json")
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for item in payload["items"]:
            if fusion_weight > 0.0:
                item["fusion_score"] = (1.0 - fusion_weight) * item["clip_score"] + fusion_weight * item["dino_score"]
        sort_key = "fusion_score" if fusion_weight > 0.0 else "clip_score"
        payload["sort_by"] = sort_key
        payload["items"] = sorted(payload["items"], key=lambda item: item[sort_key], reverse=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        csv_path = os.path.join(save_dir, "scores_sorted.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = ["rank", "index_in_batch", "clip_score", "dino_score"]
            if fusion_weight > 0.0:
                header.append("fusion_score")
            writer.writerow(header)
            for rank, item in enumerate(payload["items"]):
                row = [rank, item["index"], f"{item['clip_score']:.6f}", f"{item['dino_score']:.6f}"]
                if fusion_weight > 0.0:
                    row.append(f"{item['fusion_score']:.6f}")
                writer.writerow(row)
        return (f"Saved JSON to {json_path} and CSV to {csv_path}",)


NODE_CLASS_MAPPINGS = {"CLIPMinRanker": CLIPMinRanker}
NODE_DISPLAY_NAME_MAPPINGS = {"CLIPMinRanker": "CLIP Min Ranker"}
