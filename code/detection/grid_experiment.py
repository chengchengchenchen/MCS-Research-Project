import json, shutil, random
from pathlib import Path

IMG_EXT_CANDIDATES = ['.png', '.jpg', '.jpeg', '.bmp']

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clear_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def autodetect_images(img_dir: Path):
    imgs = []
    for ext in IMG_EXT_CANDIDATES:
        imgs.extend(sorted(img_dir.glob(f"*{ext}")))
    return sorted(imgs)

def rank_items(items, mode: str, alpha: float):
    if mode == 'clip':
        scored = [(it['index'], float(it['clip_score'])) for it in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx,_ in scored]
    elif mode == 'dino':
        scored = [(it['index'], float(it['dino_score'])) for it in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx,_ in scored]
    elif mode == 'mixed':
        scored = [(it['index'], alpha*float(it['clip_score']) + (1.0 - alpha)*float(it['dino_score'])) for it in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx,_ in scored]
    else:  # 'random'
        idxs = [int(it['index']) for it in items]
        random.shuffle(idxs)
        return idxs

def process_split(root: Path, out: Path, split: str, include_real: bool, topk: int, sort_mode: str, alpha: float):
    split_root = root / split
    in_img_dir = split_root / 'original' / 'images'
    in_lbl_dir = split_root / 'original' / 'labels'
    synth_root = split_root / 'synth'

    out_img_dir = out / split / 'images'
    out_lbl_dir = out / split / 'labels'
    ensure_dir(out_img_dir); ensure_dir(out_lbl_dir)

    originals = autodetect_images(in_img_dir)
    print(f"[{split}] detected originals: {len(originals)} at {in_img_dir}")
    if len(originals) > 0:
        print('  e.g.:', originals[0].name, '...')

    num_o = num_s = 0

    for oimg in originals:
        stem = oimg.stem
        olbl = in_lbl_dir / f"{stem}.txt"

        if include_real:
            shutil.copy2(oimg, out_img_dir / oimg.name)
            shutil.copy2(olbl, out_lbl_dir / olbl.name)
            num_o += 1

        sdir = synth_root / stem
        sj = sdir / 'scores.json'
        with sj.open('r', encoding='utf-8') as f:
            scores = json.load(f)

        order = rank_items(scores['items'], sort_mode, alpha)
        chosen = order[:topk]

        for idx in chosen:
            name_wo_ext = f"{stem}_{int(idx):02d}"
            simg = None
            for ext in IMG_EXT_CANDIDATES:
                p = sdir / f"{name_wo_ext}{ext}"
                if p.exists():
                    simg = p
                    break
            if simg is None:
                continue
            slbl_name = f"{name_wo_ext}.txt"
            shutil.copy2(simg, out_img_dir / simg.name)
            shutil.copy2(olbl, out_lbl_dir / slbl_name)
            num_s += 1

    print(f"[{split}] originals added: {num_o}, synthetic added: {num_s}")
    print(f"Output images: {out_img_dir}")
    print(f"Output labels: {out_lbl_dir}")

# ======================
# Grid experiment (minimal implementation)
# - Only write CSV using res.results_dict
# Dependency: clear_dir(root) and process_split(root, out, split, include_real, topk, sort, alpha) are defined above
# ======================
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import glob

def count_images(dir_path: Path):
    """Count files under common image extensions in a directory."""
    n = 0
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        n += len(glob.glob(str(dir_path / pat)))
    return n

def main():
    # Base hyperparameters 
    ROOT = '../dataset'           # Input root directory (contains train/ and valid/)
    OUT = 'workdir'               # Output root directory (rebuilt each run)
    SPLITS = ['train', 'valid']   # Splits to process
    SORT = 'clip'                 # 'clip' | 'dino' | 'mixed' | 'random'
    ALPHA = 0                     # For mixed: score = alpha*clip + (1-alpha)*dino

    EPOCHS = 100
    IMGSZ = 416
    BATCH = 32
    WORKERS = 8                   
    SEED = 1
    PROJECT = "runs_grid"         # Training output directory
    BASE_WEIGHTS = "yolov8s.pt"   # Pretrained weights
    DATA_YAML = "data.yaml"       # Train/val config

    # Grid: INCLUDE_REAL ∈ {True, False}; TOPK ∈ {0..9}
    include_real_opts = [True, False]
    topk_opts = list(range(0, 9))

    csv_path = Path(f"grid_metrics_{SORT}_{IMGSZ}_{SEED}.csv")

    for include_real in include_real_opts:
        for topk in topk_opts:
            # 1) Rebuild OUT dataset (overwrite the same path)
            root = Path(ROOT)
            out = Path(OUT)
            clear_dir(out)
            for split in SPLITS:
                process_split(root, out, split, include_real, topk, SORT, ALPHA)

            # 2) Pre-training count check (avoid DataLoader crash due to empty set)
            n_train = count_images(out / "train" / "images")
            n_val   = count_images(out / "valid" / "images")
            if n_train == 0 or n_val == 0:
                continue

            # 3) Train
            run_name = f"ir{int(include_real)}_k{topk}_e{EPOCHS}_{IMGSZ}"
            model = YOLO(BASE_WEIGHTS)
            model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, workers=WORKERS, seed=SEED, project=PROJECT, name=run_name)

            # 4) Validate on test split
            res = model.val(data=DATA_YAML, imgsz=IMGSZ, split="test")

            # 5) Write CSV
            row = dict(getattr(res, "results_dict", {}) or {})
            row.update({
                "include_real": include_real,
                "topk": topk,
                "sort": SORT,
                "alpha": ALPHA,
                "run_name": run_name,
                "project": PROJECT,
                "n_train": n_train,
                "n_val": n_val,
            })
            header = not csv_path.exists()
            pd.DataFrame([row]).to_csv(csv_path, index=False, mode="a", header=header)

if __name__ == "__main__":
    main()
