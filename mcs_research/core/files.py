from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def list_images(image_dir: Path, *, recursive: bool = False) -> list[Path]:
    iterator: Iterable[Path] = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        [path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: str(path).lower(),
    )


def find_image(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def materialize_file(src: Path, dst: Path, mode: str = "copy") -> None:
    src = src.resolve()
    ensure_dir(dst.parent)
    if dst.exists():
        try:
            if dst.samefile(src):
                return
        except OSError:
            pass
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to hardlink {src} -> {dst}. Use image_storage='copy' if the "
                "source and destination are on different volumes."
            ) from exc
        return
    raise ValueError(f"Unsupported materialization mode: {mode}")


def write_image_list(path: Path, image_paths: Sequence[Path]) -> None:
    ensure_dir(path.parent)
    lines = [str(image_path.resolve()) for image_path in image_paths]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
