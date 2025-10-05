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


# ComfyUI IMAGE tensor convention: [B, H, W, C], float32 in [0, 1]
class CLIPMinRanker:
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
                # Set fusion_weight=0 to rank by CLIP only
                "fusion_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Available DINOv2 model options
                "dino_model_name": ([
                    "vit_base_patch14_dinov2.lvd142m",
                    "vit_small_patch14_dinov2.lvd142m",
                    "vit_large_patch14_dinov2.lvd142m"
                ], {"default": "vit_base_patch14_dinov2.lvd142m"}),
                # Pooling strategy for DINO global embedding
                "dino_pool": (["cls", "mean", "gap", "auto"], {"default": "auto"}),
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
            enc = F.normalize(enc.float(), dim=-1)
            feats.append(enc)
        return torch.cat(feats, dim=0)  # [N, D]

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

        pool options:
          - "auto": prefer dict keys then reduce shape automatically
          - "cls": take CLS token if available, otherwise fallback
          - "mean": mean over token dimension if 3D, otherwise GAP if 4D
          - "gap": global average pool over HxW if 4D

        Supported inputs:
          - dict of tensors from forward_features
          - 3D tensor [B, T, D] token embeddings
          - 4D tensor [B, C, H, W] spatial feature maps
          - 2D tensor [B, D] already global
        """
        def gap_4d(x: torch.Tensor) -> torch.Tensor:
            return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C]

        if isinstance(feat, dict):
            # prefer stable global representations if provided
            candidate = None
            for k in ["x_norm_clstoken", "pooled", "cls_token"]:
                if k in feat and isinstance(feat[k], torch.Tensor):
                    candidate = feat[k]
                    break
            if candidate is None:
                # fallback to last tensor-like value
                vals = [v for v in feat.values() if isinstance(v, torch.Tensor)]
                if len(vals) == 0:
                    raise RuntimeError("DINO forward_features returned an unsupported dict structure.")
                candidate = vals[-1]
            x = candidate
        else:
            x = feat

        if x.dim() == 2:
            # already [B, D]
            return x

        if x.dim() == 3:
            # [B, T, D] token embeddings
            if pool == "cls":
                # Use CLS token if there are multiple tokens
                return x[:, 0] if x.size(1) > 1 else x.squeeze(1)
            elif pool == "mean":
                return x.mean(dim=1)
            elif pool == "gap":
                # GAP does not apply to token dimension; fallback to mean
                return x.mean(dim=1)
            else:
                # auto: prefer CLS if seems present, else mean
                return x[:, 0] if x.size(1) > 1 else x.mean(dim=1)

        if x.dim() == 4:
            # [B, C, H, W] spatial maps
            if pool in ("gap", "auto", "mean"):
                return gap_4d(x)
            elif pool == "cls":
                # No CLS in spatial maps; fallback to GAP
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
            fusion_weight: float = 0.0,
            dino_model_name: str = "vit_base_patch14_dinov2.lvd142m",
            dino_pool: str = "auto") -> Tuple[str]:

        assert source_image.ndim == 4 and source_image.shape[0] == 1, "source_image must be a single image batch"
        assert candidate_images.ndim == 4 and candidate_images.shape[0] >= 1, "candidate_images must be a batch"

        # Load models if not yet initialized
        self._lazy_model()
        self._lazy_dino(dino_model_name)

        os.makedirs(save_dir, exist_ok=True)

        # Convert tensors to PIL format
        src_pil = self._tensor_to_pil(source_image[0])
        cand_pils = [self._tensor_to_pil(candidate_images[i]) for i in range(candidate_images.shape[0])]

        # Encode and compute similarity scores
        with torch.no_grad():
            # CLIP embeddings
            src_clip = self._encode_images_clip([src_pil], batch_size=1)[0]                 # [D_c]
            cand_clip = self._encode_images_clip(cand_pils, batch_size=batch_size)          # [N, D_c]
            clip_scores = (cand_clip @ src_clip).cpu().numpy().tolist()                     # N

            # DINOv2 embeddings with robust global pooling
            src_dino = self._encode_images_dino([src_pil], batch_size=1, pool=dino_pool)[0] # [D_d]
            cand_dino = self._encode_images_dino(cand_pils, batch_size=batch_size, pool=dino_pool)  # [N, D_d]
            dino_scores = (cand_dino @ src_dino).cpu().numpy().tolist()                     # N

        # Build per-image records
        items: List[Dict[str, Any]] = []
        for i in range(len(clip_scores)):
            rec: Dict[str, Any] = {
                "index": int(i),
                "clip_score": float(clip_scores[i]),
                "dino_score": float(dino_scores[i]),
            }
            if fusion_weight > 0.0:
                w = float(fusion_weight)
                rec["fusion_score"] = float((1.0 - w) * rec["clip_score"] + w * rec["dino_score"])
            items.append(rec)

        # Sort by fusion score if used, otherwise by CLIP score
        sort_key = "fusion_score" if fusion_weight > 0.0 else "clip_score"
        items_sorted = sorted(items, key=lambda x: x[sort_key], reverse=True)

        # Save JSON metadata
        json_path = os.path.join(save_dir, "scores.json")
        meta: Dict[str, Any] = {
            "clip_model": "ViT-B-32 openai",
            "dino_model": dino_model_name,
            "dino_pool": dino_pool,
            "num_candidates": len(items),
            "sort_by": sort_key,
            "items": items
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save CSV for quick inspection
        csv_path = os.path.join(save_dir, "scores_sorted.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["rank", "index_in_batch", "clip_score", "dino_score"]
            if fusion_weight > 0.0:
                header.append("fusion_score")
            writer.writerow(header)
            for r, rec in enumerate(items_sorted):
                row = [r, rec["index"], f"{rec['clip_score']:.6f}", f"{rec['dino_score']:.6f}"]
                if fusion_weight > 0.0:
                    row.append(f"{rec['fusion_score']:.6f}")
                writer.writerow(row)

        summary = f"Saved JSON to {json_path} and CSV to {csv_path}"
        return (summary,)


NODE_CLASS_MAPPINGS = {
    "CLIPMinRanker": CLIPMinRanker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPMinRanker": "CLIP Min Ranker"
}
