from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _imports():
    import torch
    import torch.nn.functional as F
    import open_clip
    import timm
    from timm.data import create_transform, resolve_model_data_config

    return torch, F, open_clip, timm, create_transform, resolve_model_data_config


def _raw(text: str) -> str:
    return text


def _a(text: str) -> str:
    return "a " + text


def _picture(text: str) -> str:
    return "picture of a " + text


PROMPT_TEMPLATES = {"csl": _raw, "csl_a": _a, "csl_p": _picture}


class CalculateScore:
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
                "dino_model_name": (
                    [
                        "vit_base_patch14_dinov2.lvd142m",
                        "vit_small_patch14_dinov2.lvd142m",
                        "vit_large_patch14_dinov2.lvd142m",
                    ],
                    {"default": "vit_base_patch14_dinov2.lvd142m"},
                ),
                "dino_pool": (["cls", "mean", "gap", "auto"], {"default": "auto"}),
                "label_file": ("STRING", {"multiline": False, "default": ""}),
                "data_yaml": ("STRING", {"multiline": False, "default": ""}),
                "prompt_key": (["csl", "csl_a", "csl_p"], {"default": "csl_p"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_summary",)
    FUNCTION = "run"
    CATEGORY = "similarity"

    def __init__(self):
        torch, *_ = _imports()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.dino_model = None
        self.dino_transform = None
        self.dino_model_name_cached = None

    def _lazy_clip(self):
        _, _, open_clip, *_ = _imports()
        if self.model is None:
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=self.device)
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def _lazy_dino(self, dino_model_name: str):
        _, _, _, timm, create_transform, resolve_model_data_config = _imports()
        if self.dino_model is None or self.dino_model_name_cached != dino_model_name:
            model = timm.create_model(dino_model_name, pretrained=True)
            model.eval().to(self.device)
            self.dino_model = model
            self.dino_transform = create_transform(**resolve_model_data_config(model), is_training=False)
            self.dino_model_name_cached = dino_model_name

    @staticmethod
    def _tensor_to_pil(img_tensor) -> Image.Image:
        array = (img_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

    def _encode_images_clip(self, pil_list, batch_size: int):
        torch, F, *_ = _imports()
        feats = []
        for start in range(0, len(pil_list), batch_size):
            batch = pil_list[start : start + batch_size]
            batch_t = torch.stack([self.preprocess(image) for image in batch], dim=0).to(self.device)
            feats.append(F.normalize(self.model.encode_image(batch_t).float(), dim=-1))
        return torch.cat(feats, dim=0)

    def _encode_text_clip(self, prompts: list[str]):
        _, F, *_ = _imports()
        tokens = self.tokenizer(prompts).to(self.device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)

    def _dino_global_rep(self, feat: Any, pool: str = "auto"):
        torch, F, *_ = _imports()

        def gap_4d(x):
            return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        if isinstance(feat, dict):
            tensors = [feat[key] for key in ("x_norm_clstoken", "pooled", "cls_token") if key in feat]
            x = tensors[0] if tensors else [value for value in feat.values() if hasattr(value, "dim")][-1]
        else:
            x = feat
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x[:, 0] if pool in {"cls", "auto"} and x.size(1) > 1 else x.mean(dim=1)
        if x.dim() == 4:
            return gap_4d(x)
        raise RuntimeError(f"Unsupported DINO feature shape: {tuple(x.shape)}")

    def _encode_images_dino(self, pil_list, batch_size: int, pool: str):
        torch, F, *_ = _imports()
        feats = []
        for start in range(0, len(pil_list), batch_size):
            batch = pil_list[start : start + batch_size]
            batch_t = torch.stack([self.dino_transform(image) for image in batch], dim=0).to(self.device)
            feats.append(F.normalize(self._dino_global_rep(self.dino_model.forward_features(batch_t), pool=pool).float(), dim=-1))
        return torch.cat(feats, dim=0)

    @staticmethod
    def _load_class_names(yaml_path: str) -> list[str]:
        if yaml is None:
            raise RuntimeError("PyYAML is required for bbox prompt scoring.")
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        names = data.get("names")
        if isinstance(names, list):
            return [str(item) for item in names]
        if isinstance(names, dict):
            keys = [int(key) for key in names]
            return [str(names.get(i, names.get(str(i), f"class_{i}"))) for i in range(max(keys) + 1)]
        raise ValueError("'names' in data.yaml must be a list or dict.")

    @staticmethod
    def _parse_yolo_labels(label_path: str) -> list[dict[str, float]]:
        entries = []
        for raw_line in Path(label_path).read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            try:
                entries.append({"cid": int(float(parts[0])), "xc": float(parts[1]), "yc": float(parts[2]), "w": float(parts[3]), "h": float(parts[4])})
            except ValueError:
                continue
        return entries

    @staticmethod
    def _norm_to_xyxy(entry: dict[str, float], img_w: int, img_h: int):
        bx = entry["xc"] * img_w
        by = entry["yc"] * img_h
        bw = entry["w"] * img_w
        bh = entry["h"] * img_h
        return max(0.0, bx - bw / 2.0), max(0.0, by - bh / 2.0), min(float(img_w), bx + bw / 2.0), min(float(img_h), by + bh / 2.0)

    def run(
        self,
        source_image,
        candidate_images,
        save_dir: str,
        batch_size: int = 8,
        dino_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        dino_pool: str = "auto",
        label_file: str = "",
        data_yaml: str = "",
        prompt_key: str = "csl_p",
    ):
        torch, *_ = _imports()
        self._lazy_clip()
        self._lazy_dino(dino_model_name)
        os.makedirs(save_dir, exist_ok=True)
        source_pil = self._tensor_to_pil(source_image[0])
        candidate_pils = [self._tensor_to_pil(candidate_images[index]) for index in range(candidate_images.shape[0])]

        with torch.no_grad():
            source_clip = self._encode_images_clip([source_pil], 1)[0]
            candidate_clip = self._encode_images_clip(candidate_pils, batch_size)
            clip_scores = (candidate_clip @ source_clip).cpu().numpy().tolist()
            source_dino = self._encode_images_dino([source_pil], 1, dino_pool)[0]
            candidate_dino = self._encode_images_dino(candidate_pils, batch_size, dino_pool)
            dino_scores = (candidate_dino @ source_dino).cpu().numpy().tolist()

        bbox_by_index: dict[int, list[dict[str, Any]]] = {}
        if label_file and data_yaml:
            class_names = self._load_class_names(data_yaml)
            labels = self._parse_yolo_labels(label_file)
            formatter = PROMPT_TEMPLATES.get(prompt_key, _raw)
            text_cache = {}
            for label in labels:
                cid = int(label["cid"])
                if cid not in text_cache:
                    name = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
                    prompt = formatter(str(name))
                    text_cache[cid] = {"name": name, "prompt": prompt, "feat": self._encode_text_clip([prompt])[0]}
            for image_index, image in enumerate(candidate_pils):
                boxes = []
                for bbox_index, label in enumerate(labels):
                    cid = int(label["cid"])
                    x1, y1, x2, y2 = self._norm_to_xyxy(label, *image.size)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop_feat = self._encode_images_clip([image.crop((x1, y1, x2, y2))], 1)[0]
                    cache = text_cache[cid]
                    boxes.append(
                        {
                            "bbox_index": bbox_index,
                            "class_id": cid,
                            "class_name": cache["name"],
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "prompt": cache["prompt"],
                            "clip_score": float((crop_feat @ cache["feat"]).item()),
                        }
                    )
                bbox_by_index[image_index] = boxes

        items = [
            {"index": int(index), "clip_score": float(clip_scores[index]), "dino_score": float(dino_scores[index]), "bboxes": bbox_by_index.get(index, [])}
            for index in range(len(clip_scores))
        ]
        json_path = os.path.join(save_dir, "scores.json")
        payload = {
            "clip_model": "ViT-B-32 openai",
            "dino_model": dino_model_name,
            "dino_pool": dino_pool,
            "num_candidates": len(items),
            "items": items,
            "bbox_prompt_key": prompt_key,
            "has_bboxes": bool(label_file and data_yaml),
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        csv_path = os.path.join(save_dir, "scores.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index_in_batch", "clip_score", "dino_score"])
            for item in items:
                writer.writerow([item["index"], f"{item['clip_score']:.6f}", f"{item['dino_score']:.6f}"])
        return (f"Saved JSON to {json_path} and CSV to {csv_path}",)


NODE_CLASS_MAPPINGS = {"CalculateScore": CalculateScore}
NODE_DISPLAY_NAME_MAPPINGS = {"CalculateScore": "Calculate Score"}
