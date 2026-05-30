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

## LoRA Training with OneTrainer

LoRA training is done manually in the OneTrainer GUI. This repository prepares the captioned image folders and provides
a preset template, but it does not launch OneTrainer training automatically.

Scope:

- Train one LoRA per dataset.
- The 8 datasets are `voc`, `apex_game`, `aquarium`, `cotton`, `mri_image`, `road_traffic`, `robomaster`, and
  `underwater`.
- Use only the pipeline-materialized train split for LoRA training.
- The LoRA train split must contain 200 images.
- Caption sidecar files are required: every `*.jpg`/`*.png` in the train folder must have a matching `*.txt` caption.

Expected LoRA train folder:

```text
workdir/<dataset-run>/filtered/train/images/
  image_0001.jpg
  image_0001.txt
  ...
```

Before opening OneTrainer, verify the train folder has 200 images and paired captions:

```bash
find workdir/<dataset-run>/filtered/train/images -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
find workdir/<dataset-run>/filtered/train/images -type f -iname '*.txt' | wc -l
```

OneTrainer setup:

1. Install OneTrainer separately from `https://github.com/Nerogar/OneTrainer`.
2. Start the OneTrainer GUI with `start-ui.bat` on Windows or `./start-ui.sh` on Linux/macOS.
3. Copy the preset template into OneTrainer:

```bash
cp onetrainer_presets/sd35_large_turbo_lora_200_train_with_caption.json /path/to/OneTrainer/training_presets/<dataset>.json
```

Windows PowerShell example:

```powershell
$dataset = "voc"
Copy-Item onetrainer_presets\sd35_large_turbo_lora_200_train_with_caption.json "E:\a\OneTrainer\training_presets\$dataset.json"
```

In the OneTrainer GUI:

- Load the copied preset.
- Set `concept_file_name` to a concept JSON for this dataset, for example
  `training_concepts/<dataset>_concepts.json`.
- In that concept JSON, use one enabled standard concept whose path is the 200-image pipeline train folder:
  `workdir/<dataset-run>/filtered/train/images`.
- Keep prompt source as sample/sidecar caption so OneTrainer reads the paired `*.txt` files.
- Set `output_model_destination` to the unified LoRA name:
  `models/<dataset>_sd35_large_turbo_lora.safetensors`.

The preset template is based on `voc-10-with.json` and preserves these training settings:

- base model: `stabilityai/stable-diffusion-3.5-large-turbo`
- model type: `STABLE_DIFFUSION_35`
- training method: `LORA`
- epochs: `400`
- batch size: `16`
- gradient accumulation: `1`
- learning rate: `0.0003`
- scheduler: `COSINE`
- warmup steps: `200`
- optimizer: `ADAMW`
- weight decay: `0.01`
- resolution: `512`
- aspect ratio bucketing: enabled
- latent caching: enabled
- gradient checkpointing: `ON`
- train device: `cuda`
- temp device: `cpu`
- train dtype: `FLOAT_16`
- fallback train dtype: `BFLOAT_16`
- output dtype: `FLOAT_32`
- output format: `SAFETENSORS`
- LoRA rank: `16`
- LoRA alpha: `1.0`
- validation: disabled
- TensorBoard: enabled
- save every: `100` epochs

After training, use the generated `models/<dataset>_sd35_large_turbo_lora.safetensors` in the ComfyUI workflow for that
dataset.

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
