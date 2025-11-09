# Generation Workflow

## Overview
`comfyUI_script.py` automates batch image generation by driving a local ComfyUI backend at `http://127.0.0.1:8000`. It loads a prebuilt workflow, rewrites node inputs for each dataset image, and waits for ComfyUI to finish before moving on.

## Prerequisites
- Python 3.8+ with `requests` and `tqdm`.
- A ComfyUI server running locally and exposing the REST API.
- A workflow JSON file (default `canny-clip.json`) that matches the node IDs hard-coded in the script.

## Configuration
Edit the constants at the top of `comfyUI_script.py` to fit its environment:
- `PROMPT_PATH`: path to the ComfyUI workflow JSON.
- `INPUT_DIR`: directory containing input `.jpg`, `.png`, or `.jpeg` files.
- `OUTPUT_DIR`: root folder where per-image results are stored.
- `NODE_ID_*`: node identifiers inside the workflow; keep them aligned with `PROMPT_PATH` workflow.

## Processing Steps
1. Resolve absolute paths for the workflow file, dataset directory, and output root.
2. Load the workflow once and clone it for each image.
3. Inject the current image path into the LoadImage node.
4. Point CLIP and CustomSaveImage nodes to a dedicated subfolder named after the image basename.
5. Submit the prompt to the ComfyUI REST API and poll `/history/{prompt_id}` until the run completes.
6. Save the generated output as `<basename>.*` and the source image as `_<basename>.*` inside `OUTPUT_DIR/<basename>/`.

## Output Layout
```
OUTPUT_DIR/
    <image_basename>/
        <image_basename>.*      # generated result
        _<image_basename>.*     # saved source image
```



TODO: Replace hard-coded constants with CLI arguments for repeated runs.
