## Overview

- This folder contains two ComfyUI custom nodes for batch image saving and similarity-based ranking.
- Image tensors follow ComfyUI’s convention: `IMAGE` is a float32 tensor in `[0,1]` with shape `[B, H, W, C]`.

## Deployment

copy this folder to `/path/to/ComfyUI/custom_nodes` and restart ComfyUI



## Custom Save Image (Batch)

- File: `code/custom_node/custom_save_image.py`

- Display name: `Custom Save Image (Batch)`

- Class: `CustomSaveImage`

- Purpose
  - Save a batch of images from a ComfyUI `IMAGE` tensor to disk as .PNG files.

- Inputs (required)
  - `image` (`IMAGE`): Input image tensor `[B, H, W, C]`, float32 `[0,1]`.
  - `save_dir` (`STRING`, default `outputs/custom`): Directory to write outputs.
  - `file_name` (`STRING`, default `sample`): Base filename prefix.

- Outputs
  - No returned tensors (output node). Side effect: writes PNG files to `save_dir`.
  - Name pattern: `<file_name>_<index>.png` where `index` is zero-padded per batch item.

  

## CLIP Min Ranker

- File: `code/custom_node/clip_min_ranker.py`
- Display name: `CLIP Min Ranker`
- Class: `CLIPMinRanker`

- Purpose
  - Compute similarity scores between a single `source_image` and a batch of `candidate_images` using CLIP and DINOv2 features.
  - Saves a sortable CSV and a JSON summary of scores; returns a string summary path.

- Inputs
  - Required
    - `source_image` (`IMAGE`): Single-image batch `[1, H, W, C]` as float32 `[0,1]`.
    - `candidate_images` (`IMAGE`): Batch of candidate images `[N, H, W, C]` as float32 `[0,1]`.
    - `save_dir` (`STRING`, default `outputs/clip_scores`): Directory to save results.
  - Optional
    - `batch_size` (`INT`, default `8`, min `1`, max `128`): Batch size for model encodings.
    - `fusion_weight` (`FLOAT`, default `0.0`, range `[0.0, 1.0]`):
      - `0.0`: rank by CLIP only.
      - `>0.0`: linear fusion `(1-w)*CLIP + w*DINO`.
    - `dino_model_name` (choice, default `vit_base_patch14_dinov2.lvd142m`): One of
      - `vit_small_patch14_dinov2.lvd142m`
      - `vit_base_patch14_dinov2.lvd142m`
      - `vit_large_patch14_dinov2.lvd142m`
    - `dino_pool` (choice, default `auto`): `cls` | `mean` | `gap` | `auto`. Controls global embedding pooling for DINOv2 features.

- Outputs
  - Returns: `(result_summary: STRING,)` — a short message containing saved file paths.
  - Side effects in `save_dir`:
    - `scores.json`: Metadata including model names, pooling mode, sort key, and per-candidate scores.
    - `scores_sorted.csv`: Tabular scores sorted by the effective ranking key.

- Ranking details
  - CLIP: `open_clip` ViT-B/32 (OpenAI) image encoder with cosine-normalized embeddings.
  - DINOv2: Uses `timm` with robust pooling over `forward_features` outputs; embeddings are cosine-normalized.
  - Sort key: `fusion_score` if `fusion_weight > 0.0`, else `clip_score`.
