import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

# === Basic configuration ===
# TODO: Transfer hardcode path to paras input in cmd
API = "http://127.0.0.1:8000"                   # ComfyUI backend API
PROMPT_PATH = "new_clip_calculate_2.json"       # Workflow file path
INPUT_DIR = "../dataset/train/original/images"  # Input dataset folder
LABEL_DIR = "../dataset/train/original/labels"  # YOLO label folder
DATA_YAML_PATH = "../detection/data.yaml"       # data.yaml path for labels
OUTPUT_DIR = "outputs/clip_calculate_train_2"           # Root output folder
BATCH_SIZE = 8
CLIENT_ID = "python"

# === Node IDs in ComfyUI workflow ===
NODE_ID_LOADIMAGE = "7"               # LoadImage node ID
NODE_ID_SAVEIMAGE = "30"              # CustomSaveImage node ID for generated images
NODE_ID_LABEL_PATH = "40"             # PrimitiveString node for YOLO label file
NODE_ID_DATA_YAML = "41"              # PrimitiveString node for data.yaml
NODE_ID_SAVE_DIR = "42"               # PrimitiveString node for save_dir
NODE_ID_BATCH_SIZE = "43"             # PrimitiveInt node for batch size

def load_prompt(path: str) -> Dict[str, Any]:
    """Load a JSON workflow file and return its node dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j.get("prompt", j)

def submit(prompt: Dict[str, Any]) -> str:
    """Submit a prompt (workflow instance) to the ComfyUI backend."""
    r = requests.post(f"{API}/prompt", json={"prompt": prompt, "client_id": CLIENT_ID})
    r.raise_for_status()
    return r.json()["prompt_id"]

def wait_done(pid: str, timeout: float = 600.0, poll: float = 0.5) -> Dict[str, Any]:
    """Poll the /history API until the given prompt ID is finished."""
    t0 = time.time()
    while True:
        r = requests.get(f"{API}/history/{pid}")
        if r.ok:
            hist = r.json()
            if pid in hist:
                return hist[pid]
        if time.time() - t0 > timeout:
            raise TimeoutError("Prompt execution timed out.")
        time.sleep(poll)

def main():
    # === Resolve all paths to absolute paths ===
    prompt_path = Path(PROMPT_PATH).resolve()
    input_dir = Path(INPUT_DIR).resolve()
    label_dir = Path(LABEL_DIR).resolve()
    data_yaml = Path(DATA_YAML_PATH).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()

    print(f"Input directory: {input_dir}")
    print(f"Label directory: {label_dir}")
    print(f"data.yaml: {data_yaml}")
    print(f"Output root: {output_dir}")

    base_prompt = load_prompt(str(prompt_path))
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    # Inject static values that stay the same for every run
    base_prompt[NODE_ID_DATA_YAML]["inputs"]["value"] = str(data_yaml.as_posix())
    base_prompt[NODE_ID_BATCH_SIZE]["inputs"]["value"] = BATCH_SIZE
    os.makedirs(output_dir, exist_ok=True)

    # === Collect all images from dataset ===
    imgs = [p for p in sorted(os.listdir(input_dir)) if p.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not imgs:
        print("No images found in", input_dir)
        return

    # === Iterate through dataset images with tqdm progress bar ===
    for i, img_name in enumerate(tqdm(imgs, desc="Processing images", unit="img"), start=1):
        img_path = (input_dir / img_name).resolve()
        base_name = Path(img_name).stem

        label_path = (label_dir / f"{base_name}.txt").resolve()
        if not label_path.exists():
            tqdm.write(f"[{i:02d}] Missing label file for {img_name}, expected {label_path}")
            continue

        prompt = json.loads(json.dumps(base_prompt))

        prompt[NODE_ID_LOADIMAGE]["inputs"]["image"] = str(img_path.as_posix())
        prompt[NODE_ID_LABEL_PATH]["inputs"]["value"] = str(label_path.as_posix())

        run_dir = (output_dir / base_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generated image saver
        prompt[NODE_ID_SAVE_DIR]["inputs"]["value"] = str(run_dir.as_posix())
        prompt[NODE_ID_SAVEIMAGE]["inputs"]["file_name"] = base_name

        pid = submit(prompt)
        tqdm.write(f"[{i:02d}] pid={pid}, image={img_path.name}")
        _ = wait_done(pid)
        tqdm.write(f"[{i:02d}] saved under {run_dir}")
        time.sleep(0.2)

if __name__ == "__main__":
    main()
