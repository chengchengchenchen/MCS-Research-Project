# Thesis Reproduction YOLO Research Pipeline

This repository is the standalone source tree for reproducing the thesis YOLO data-generation experiments. It
does not depend on the older project `code/` folder or on datasets outside this repository. A fresh clone starts
with no dataset: downloaded data, generated candidates, training runs, weights, and caches are written under
ignored local directories.

## Pipeline

The reproducible experiment chain is:

1. Prepare a YOLO-format dataset from COCO-FSOD, Pascal VOC, or RF100-style YOLO exports.
2. Optionally prepare LoRA inputs and image captions.
3. Optionally run ComfyUI generation for synthetic candidates.
4. Filter synthetic candidates with the preserved CLIP/DINO/bbox scoring policy.
5. Train and evaluate Ultralytics YOLO.

Default output locations:

- datasets and downloads: `data/`
- intermediate datasets and generated candidates: `workdir/`
- YOLO runs: `workdir/<dataset>/yolo_runs/`

These directories are ignored by git.

## Environment

Recommended base environment:

- Python 3.10-3.12 for training/generation compatibility with PyTorch and Ultralytics.
- Linux/macOS bash, WSL bash, or Windows Git Bash for the end-to-end scripts.
- CUDA GPU recommended for YOLO training and ComfyUI generation. CPU is sufficient for tests and `--dry-run`.
- ComfyUI is required only when `RUN_GENERATION=1`.
- Ollama is required only for caption generation in `prepare_lora.py` when captioning is enabled.

Create an environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches your CUDA driver before or after `requirements.txt`,
following the official PyTorch selector. The repository code does not pin a CUDA wheel because the correct wheel
depends on the machine.

## Optional Services

ComfyUI generation:

- Start ComfyUI separately and keep its API reachable, usually at `http://127.0.0.1:8000`.
- Copy or symlink `comfyui_custom_nodes/` into your ComfyUI custom nodes directory if the workflow uses these
  nodes.
- Provide the workflow JSON with `COMFYUI_WORKFLOW=/path/to/workflow.json` or `COMFYUI_PROMPT_PATH=...`.
- Enable generation with `RUN_GENERATION=1 INCLUDE_SYNTH=1`.

Ollama captioning:

- Default endpoint: `http://127.0.0.1:11434`.
- Default model in code: `qwen3-vl:32b`.
- To skip captioning, set `skip_caption: true` in the `prepare_lora.py` config.

## Dataset Inputs

The scripts assume `data/` is empty unless you provide local data.

Pascal VOC:

- `bash bash/run_voc.sh`
- Downloads VOC2007 trainval/test and VOC2012 trainval into `data/VOCdevkit/`.

COCO-FSOD:

- `bash bash/run_coco_fsod.sh`
- Downloads COCO 2017 images/annotations.
- Requires the experiment-specific few-shot annotation JSON:

```bash
FSOD_JSON_URL=https://example/path/5shot_novel.json bash bash/run_coco_fsod.sh
# or
FSOD_JSON_PATH=/absolute/path/to/5shot_novel.json bash bash/run_coco_fsod.sh
```

RF100-style datasets:

- Scripts first use `data/rf100/<dataset>/` if it already contains YOLO `train/valid/test` splits.
- Otherwise provide a zip URL:

```bash
RF100_AQUARIUM_ZIP_URL=https://example/aquarium-yolov8.zip bash bash/run_rf100_aquarium.sh
# or one shared URL variable for a single run
RF100_ZIP_URL=https://example/dataset-yolov8.zip bash bash/run_rf100_aquarium.sh
```

- Roboflow downloads are also supported:

```bash
ROBOFLOW_API_KEY=... ROBOFLOW_WORKSPACE=... ROBOFLOW_VERSION=... bash bash/run_rf100_aquarium.sh
```

Dataset-specific overrides are available as `RF100_<DATASET>_API_KEY`, `RF100_<DATASET>_WORKSPACE`,
`RF100_<DATASET>_PROJECT`, and `RF100_<DATASET>_VERSION`.

## End-to-End Scripts

Run from the repository root:

```bash
bash bash/run_voc.sh
bash bash/run_coco_fsod.sh
bash bash/run_rf100_apex_game.sh
bash bash/run_rf100_aquarium.sh
bash bash/run_rf100_cotton.sh
bash bash/run_rf100_mri_image.sh
bash bash/run_rf100_road_traffic.sh
bash bash/run_rf100_robomaster.sh
bash bash/run_rf100_underwater.sh
```

Useful controls:

```bash
DRY_RUN=1 bash bash/run_voc.sh
SEED=1 EPOCHS=100 IMG_SIZE=416 BATCH=32 WEIGHTS=yolov8s.pt bash bash/run_rf100_aquarium.sh
RUN_GENERATION=1 INCLUDE_SYNTH=1 COMFYUI_WORKFLOW=workflow.json bash bash/run_coco_fsod.sh
```

`DRY_RUN=1` builds filtered datasets and YOLO train/eval arguments without launching training.

## Python CLIs

The bash scripts generate JSON configs and call these thin CLIs:

```bash
python scripts/prepare_dataset.py --config path/to/prepare_dataset.json
python scripts/prepare_lora.py --config path/to/prepare_lora.json
python scripts/generate_comfyui.py --config path/to/generate_comfyui.json
python scripts/filter_dataset.py --config path/to/filter_dataset.json
python scripts/train_yolo.py --config path/to/train_yolo.json --dry-run
python scripts/run_pipeline.py --config path/to/pipeline.json --dry-run-train
```

## Scientific Defaults

Synthetic filtering defaults are preserved:

- `synth_keep_ratio`: `0.3`
- `synth_keep_topk`: `-1`
- `synth_min_keep`: `0`
- `filter_quantile`: `25.0`
- weights: CLIP `0.2`, DINO `0.5`, bbox mean `0.25`, bbox min `0.05`
- `sample_seed`: `1`

Expected generated candidate layout:

```text
synth_root/
  <stem>/
    <stem>_00.png
    <stem>_01.png
    scores.json
```

`scores.json` keeps `items[*].index`, `clip_score`, `dino_score`, and optional `bboxes[*].clip_score` fields.

## Verification

Core checks:

```bash
python -m compileall mcs_research scripts comfyui_custom_nodes tests
python -m pytest tests
python scripts/prepare_dataset.py --help
python scripts/train_yolo.py --help
```

Bash syntax checks:

```bash
bash -n bash/lib/pipeline_common.sh
bash -n bash/run_voc.sh
bash -n bash/run_coco_fsod.sh
bash -n bash/run_rf100_aquarium.sh
```
