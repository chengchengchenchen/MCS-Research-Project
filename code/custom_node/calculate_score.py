# file: custom_nodes/clip_min_ranker.py
import os
import json
import csv
from typing import Tuple, List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import open_clip  # pip install open_clip_torch
import timm       # pip install timm
from timm.data import resolve_model_data_config, create_transform
import yaml

# ====== Prompt Engineering templates (参考 YOLOLabelToPrompt) ======
RAW_PROMPT = lambda prompt: prompt
A_PROMPT = lambda prompt: "a " + prompt
P_PROMPT = lambda prompt: "picture of a " + prompt

PROMPT_TEMPLATES = {
    "csl": RAW_PROMPT,   # cat
    "csl_a": A_PROMPT,   # a cat
    "csl_p": P_PROMPT,   # picture of a cat
}


# ComfyUI IMAGE tensor convention: [B, H, W, C], float32 in [0, 1]
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
                # Available DINOv2 model options
                "dino_model_name": ([
                    "vit_base_patch14_dinov2.lvd142m",
                    "vit_small_patch14_dinov2.lvd142m",
                    "vit_large_patch14_dinov2.lvd142m"
                ], {"default": "vit_base_patch14_dinov2.lvd142m"}),
                # Pooling strategy for DINO global embedding
                "dino_pool": (["cls", "mean", "gap", "auto"], {"default": "auto"}),
                # YOLO 信息（bbox + 文本 prompt 都来自这两个文件）
                "label_file": ("STRING", {"multiline": False, "default": ""}),
                "data_yaml": ("STRING", {"multiline": False, "default": ""}),
                # 使用哪种 prompt 模板
                "prompt_key": (["csl", "csl_a", "csl_p"], {"default": "csl_p"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_summary",)
    FUNCTION = "run"
    CATEGORY = "similarity"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CLIP lazy handles
        self.model = None
        self.preprocess = None
        self.tokenizer = None

        # DINOv2 lazy handles
        self.dino_model = None
        self.dino_transform = None
        self.dino_model_name_cached = None

    # ---------- CLIP ----------
    def _lazy_model(self):
        """Load CLIP model and preprocess only once."""
        if self.model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device=self.device
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @staticmethod
    def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
        """Convert a float32 tensor [H, W, C] in [0, 1] to a PIL image."""
        arr = (img_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    @torch.no_grad()
    def _encode_images_clip(self, pil_list: List[Image.Image], batch_size: int) -> torch.Tensor:
        """Encode a list of PIL images into CLIP embeddings [N, D]."""
        feats = []
        for i in range(0, len(pil_list), batch_size):
            batch = pil_list[i:i+batch_size]
            batch_t = torch.stack([self.preprocess(im) for im in batch], dim=0).to(self.device)
            enc = self.model.encode_image(batch_t)
            enc = F.normalize(enc.float(), dim=-1)  # cosine-ready
            feats.append(enc)
        return torch.cat(feats, dim=0)  # [N, D]

    @torch.no_grad()
    def _encode_text_clip(self, prompts: List[str]) -> torch.Tensor:
        """Encode a list of text prompts into CLIP embeddings [M, D]."""
        tokens = self.tokenizer(prompts).to(self.device)
        txt = self.model.encode_text(tokens)
        return F.normalize(txt.float(), dim=-1)

    @staticmethod
    def _load_class_names(yaml_path: str) -> List[str]:
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if names is None:
            raise ValueError("'names' field not found in data.yaml")
        if isinstance(names, list):
            return [str(n) for n in names]
        if isinstance(names, dict):
            max_idx = max(map(int, names.keys()))
            return [str(names.get(str(i), f"class_{i}")) for i in range(max_idx + 1)]
        raise TypeError("'names' in data.yaml must be list or dict")

    @staticmethod
    def _parse_yolo_labels(label_path: str) -> List[Dict[str, float]]:
        """
        Parse YOLO label file.
        Format per line: class x_center y_center width height (normalized).
        Returns list of dicts with cid and normalized xywh.
        """
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        entries = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    entries.append({"cid": cid, "xc": xc, "yc": yc, "w": w, "h": h})
                except ValueError:
                    continue
        return entries

    @staticmethod
    def _norm_to_xyxy(entry: Dict[str, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert normalized YOLO bbox to absolute xyxy (clamped)."""
        xc, yc, w, h = entry["xc"], entry["yc"], entry["w"], entry["h"]
        bx = xc * img_w
        by = yc * img_h
        bw = w * img_w
        bh = h * img_h
        x1 = max(0.0, bx - bw / 2.0)
        y1 = max(0.0, by - bh / 2.0)
        x2 = min(float(img_w), bx + bw / 2.0)
        y2 = min(float(img_h), by + bh / 2.0)
        return x1, y1, x2, y2

    # ---------- DINOv2 ----------
    def _lazy_dino(self, dino_model_name: str):
        """Load the DINOv2 model and its preprocessing transform when needed."""
        if self.dino_model is None or self.dino_model_name_cached != dino_model_name:
            model = timm.create_model(dino_model_name, pretrained=True)
            model.eval().to(self.device)
            data_cfg = resolve_model_data_config(model)
            transform = create_transform(**data_cfg, is_training=False)
            self.dino_model = model
            self.dino_transform = transform
            self.dino_model_name_cached = dino_model_name

    def _dino_global_rep(self, feat: Any, pool: str = "auto") -> torch.Tensor:
        """
        Convert various DINO forward_features outputs into a global embedding [B, D].
        """
        def gap_4d(x: torch.Tensor) -> torch.Tensor:
            return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C]

        if isinstance(feat, dict):
            candidate = None
            for k in ["x_norm_clstoken", "pooled", "cls_token"]:
                if k in feat and isinstance(feat[k], torch.Tensor):
                    candidate = feat[k]
                    break
            if candidate is None:
                vals = [v for v in feat.values() if isinstance(v, torch.Tensor)]
                if len(vals) == 0:
                    raise RuntimeError("DINO forward_features returned an unsupported dict structure.")
                candidate = vals[-1]
            x = candidate
        else:
            x = feat

        if x.dim() == 2:
            return x

        if x.dim() == 3:
            if pool == "cls":
                return x[:, 0] if x.size(1) > 1 else x.squeeze(1)
            elif pool == "mean":
                return x.mean(dim=1)
            elif pool == "gap":
                return x.mean(dim=1)
            else:
                return x[:, 0] if x.size(1) > 1 else x.mean(dim=1)

        if x.dim() == 4:
            if pool in ("gap", "auto", "mean"):
                return gap_4d(x)
            elif pool == "cls":
                return gap_4d(x)

        raise RuntimeError(f"Unsupported DINO feature shape: {tuple(x.shape)}")

    @torch.no_grad()
    def _encode_images_dino(self, pil_list: List[Image.Image], batch_size: int, pool: str) -> torch.Tensor:
        """Encode a list of PIL images into DINOv2 global embeddings [N, D] with robust pooling."""
        feats = []
        for i in range(0, len(pil_list), batch_size):
            batch = pil_list[i:i+batch_size]
            batch_t = torch.stack([self.dino_transform(im) for im in batch], dim=0).to(self.device)
            f = self.dino_model.forward_features(batch_t)
            rep = self._dino_global_rep(f, pool=pool)     # [B, D]
            rep = F.normalize(rep.float(), dim=-1)        # cosine-friendly
            feats.append(rep)
        return torch.cat(feats, dim=0)                    # [N, D]

    # ---------- MAIN PIPELINE ----------
    def run(self,
            source_image: torch.Tensor,
            candidate_images: torch.Tensor,
            save_dir: str,
            batch_size: int = 8,
            dino_model_name: str = "vit_base_patch14_dinov2.lvd142m",
            dino_pool: str = "auto",
            label_file: str = "",
            data_yaml: str = "",
            prompt_key: str = "csl_p") -> Tuple[str]:

        assert source_image.ndim == 4 and source_image.shape[0] == 1, "source_image must be a single image batch"
        assert candidate_images.ndim == 4 and candidate_images.shape[0] >= 1, "candidate_images must be a batch"

        # 加载模型
        self._lazy_model()
        self._lazy_dino(dino_model_name)

        use_bbox = isinstance(label_file, str) and label_file != "" and isinstance(data_yaml, str) and data_yaml != ""

        os.makedirs(save_dir, exist_ok=True)

        # Convert tensors to PIL format
        src_pil = self._tensor_to_pil(source_image[0])
        cand_pils = [self._tensor_to_pil(candidate_images[i]) for i in range(candidate_images.shape[0])]

        # 先准备一个 index -> bbox 列表 的字典（后面挂在 items 上）
        bbox_by_index: Dict[int, List[Dict[str, Any]]] = {}

        with torch.no_grad():
            # ----- CLIP: 全局图像-图像分数 -----
            cand_clip = self._encode_images_clip(cand_pils, batch_size=batch_size)          # [N, D_c]
            src_clip = self._encode_images_clip([src_pil], batch_size=1)[0]                 # [D_c]
            clip_scores = (cand_clip @ src_clip).cpu().numpy().tolist()                     # N

            # ----- DINO: 全局图像-图像分数 -----
            src_dino = self._encode_images_dino([src_pil], batch_size=1, pool=dino_pool)[0]       # [D_d]
            cand_dino = self._encode_images_dino(cand_pils, batch_size=batch_size, pool=dino_pool)  # [N, D_d]
            dino_scores = (cand_dino @ src_dino).cpu().numpy().tolist()                            # N

            # ----- CLIP: bbox + 按 class_name 生成 prompt 打分 -----
            if use_bbox:
                class_names = self._load_class_names(data_yaml)
                labels = self._parse_yolo_labels(label_file)
                if len(labels) > 0:
                    pe = PROMPT_TEMPLATES.get(prompt_key, RAW_PROMPT)

                    # 先根据 labels 构建 cid -> 文本特征 的缓存（所有图片共享）
                    text_cache: Dict[int, Dict[str, Any]] = {}
                    for entry in labels:
                        cid = entry["cid"]
                        if cid in text_cache:
                            continue

                        if 0 <= cid < len(class_names):
                            base_name = str(class_names[cid])
                        else:
                            base_name = f"class_{cid}"

                        prompt_text = pe(base_name)
                        txt_feat = self._encode_text_clip([prompt_text])[0]  # [D_c]
                        text_cache[cid] = {
                            "class_name": base_name,
                            "prompt": prompt_text,
                            "feat": txt_feat,
                        }

                    # 同一份 label_file，应用到 batch 里的每一张 candidate image
                    for image_index, base_img in enumerate(cand_pils):
                        img_w, img_h = base_img.size
                        per_image_bboxes: List[Dict[str, Any]] = []

                        for idx, entry in enumerate(labels):
                            cid = entry["cid"]
                            tc = text_cache.get(cid, None)
                            if tc is None:
                                continue

                            x1, y1, x2, y2 = self._norm_to_xyxy(entry, img_w, img_h)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            crop = base_img.crop((x1, y1, x2, y2))
                            crop_feat = self._encode_images_clip([crop], batch_size=1)[0]  # [D_c]

                            score = float((crop_feat @ tc["feat"]).item())

                            per_image_bboxes.append({
                                "bbox_index": int(idx),
                                "class_id": int(cid),
                                "class_name": tc["class_name"],
                                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                "prompt": tc["prompt"],
                                "clip_score": score,
                            })

                        bbox_by_index[image_index] = per_image_bboxes

        # Build per-image records（按原 batch 顺序保存，并挂上对应 index 的 bbox 列表）
        items: List[Dict[str, Any]] = []
        for i in range(len(clip_scores)):
            rec: Dict[str, Any] = {
                "index": int(i),                   # 在 candidate_images 里的下标
                "clip_score": float(clip_scores[i]),
                "dino_score": float(dino_scores[i]),
                # 如果该 index 有 bbox，就加上；否则为空列表
                "bboxes": bbox_by_index.get(i, []),
            }
            items.append(rec)

        # Save JSON metadata（全信息整合在一个 JSON 中）
        json_path = os.path.join(save_dir, "scores.json")
        meta: Dict[str, Any] = {
            "clip_model": "ViT-B-32 openai",
            "dino_model": dino_model_name,
            "dino_pool": dino_pool,
            "num_candidates": len(items),
            "items": items,            # 每个 item 里已经包含 bboxes
            "bbox_prompt_key": prompt_key,
            "has_bboxes": bool(use_bbox),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save CSV（同样不排序，按 index 顺序，只保存全局分数）
        csv_path = os.path.join(save_dir, "scores.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["index_in_batch", "clip_score", "dino_score"]
            writer.writerow(header)
            for rec in items:
                row = [rec["index"], f"{rec['clip_score']:.6f}", f"{rec['dino_score']:.6f}"]
                writer.writerow(row)

        summary = f"Saved JSON to {json_path} and CSV to {csv_path}"
        return (summary,)


NODE_CLASS_MAPPINGS = {
    "CalculateScore": CalculateScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CalculateScore": "Calculate Score",
}
