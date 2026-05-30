from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mcs_research.core.files import ensure_dir, find_image, list_images, load_json, materialize_file, write_json


def parse_inline_names(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        keys: list[int] = []
        for key in parsed:
            try:
                keys.append(int(key))
            except (TypeError, ValueError):
                continue
        if keys:
            return [str(parsed.get(i) or parsed.get(str(i)) or f"class_{i}") for i in range(max(keys) + 1)]
        return [str(item) for _, item in sorted(parsed.items())]
    return [str(parsed)]


def load_class_names(data_yaml: Path) -> list[str]:
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        inline_value = stripped[len("names:") :].strip()
        if inline_value:
            return parse_inline_names(inline_value)

        base_indent = len(raw_line) - len(raw_line.lstrip(" "))
        names_list: list[str] = []
        names_dict: dict[int, str] = {}
        for nested_raw in lines[index + 1 :]:
            nested_line = nested_raw.split("#", 1)[0].rstrip()
            if not nested_line.strip():
                continue
            nested_indent = len(nested_raw) - len(nested_raw.lstrip(" "))
            if nested_indent <= base_indent:
                break
            entry = nested_line.strip()
            if entry.startswith("- "):
                names_list.append(entry[2:].strip().strip("\"'"))
            elif ":" in entry:
                key_text, value_text = entry.split(":", 1)
                try:
                    names_dict[int(key_text.strip().strip("\"'"))] = value_text.strip().strip("\"'")
                except ValueError:
                    continue
        if names_list:
            return names_list
        if names_dict:
            return [names_dict.get(i, f"class_{i}") for i in range(max(names_dict) + 1)]
    return []


def write_data_yaml(
    path: Path,
    names: Sequence[str],
    *,
    train: str | Path = "train/images",
    val: str | Path = "valid/images",
    test: str | Path | None = None,
    root_path: str | Path | None = ".",
) -> None:
    ensure_dir(path.parent)
    lines: list[str] = []
    if root_path is not None:
        lines.append(f"path: {root_path}")
    lines.extend([f"train: {train}", f"val: {val}"])
    if test is not None:
        lines.append(f"test: {test}")
    lines.extend([f"nc: {len(names)}", "names:"])
    lines.extend([f"  - '{name}'" for name in names])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def yolo_label_counts(label_path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    if not label_path.exists():
        return counts
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def format_label_counts(counts: Mapping[int, int], class_names: Sequence[str]) -> str:
    tokens: list[str] = []
    for class_id in sorted(counts):
        name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        tokens.append(f"{counts[class_id]} {' '.join(str(name).split()).lower()}")
    return ", ".join(tokens)


def coco_bbox_to_yolo(bbox: Sequence[float], width: float, height: float) -> str:
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / width
    y_center = (y + h / 2.0) / height
    return f"{x_center:.6f} {y_center:.6f} {w / width:.6f} {h / height:.6f}"


def annotations_by_image(
    annotations: Iterable[Mapping[str, Any]],
    images_by_id: Mapping[int, Mapping[str, Any]],
    class_index_by_category_id: Mapping[int, int],
) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        image_id = int(ann.get("image_id", -1))
        category_id = int(ann.get("category_id", -1))
        bbox = ann.get("bbox", [])
        if image_id not in images_by_id or category_id not in class_index_by_category_id:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        grouped[image_id].append(ann)
    return grouped


def convert_coco_to_yolo(
    *,
    annotation_path: Path,
    image_root: Path,
    output_root: Path,
    split_name: str = "train",
    image_storage: str = "copy",
    overwrite_labels: bool = True,
) -> dict[str, Any]:
    data = load_json(annotation_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    names = [str(cat["name"]) for cat in categories]
    class_index_by_category_id = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    images_by_id = {int(image["id"]): image for image in images}
    grouped = annotations_by_image(annotations, images_by_id, class_index_by_category_id)

    images_out = output_root / split_name / "images"
    labels_out = output_root / split_name / "labels"
    ensure_dir(images_out)
    ensure_dir(labels_out)

    copied = 0
    labeled = 0
    for image_id, image_info in sorted(images_by_id.items()):
        file_name = str(image_info.get("file_name") or "")
        if not file_name:
            continue
        source = image_root / file_name
        if not source.exists():
            source = find_image(image_root, Path(file_name).stem)
        if source is None or not source.exists():
            raise FileNotFoundError(f"Image not found for COCO record {file_name} under {image_root}")

        dst_image = images_out / source.name
        materialize_file(source, dst_image, image_storage)
        copied += 1

        width = float(image_info.get("width", 0))
        height = float(image_info.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid size for image {file_name}: {width}x{height}")

        label_path = labels_out / f"{dst_image.stem}.txt"
        if label_path.exists() and not overwrite_labels:
            continue
        lines: list[str] = []
        for ann in grouped.get(image_id, []):
            class_index = class_index_by_category_id[int(ann["category_id"])]
            lines.append(f"{class_index} {coco_bbox_to_yolo(ann['bbox'], width, height)}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            labeled += 1

    write_data_yaml(output_root / "data.yaml", names, train=f"{split_name}/images", val=f"{split_name}/images")
    manifest = {
        "format": "coco_to_yolo",
        "annotation_path": str(annotation_path),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "split_name": split_name,
        "num_images": copied,
        "num_labeled_images": labeled,
        "num_annotations": len(annotations),
        "class_names": names,
    }
    write_json(output_root / f"{split_name}_manifest.json", manifest)
    return manifest


def summarize_yolo_split(image_dir: Path, label_dir: Path) -> dict[str, int]:
    images = list_images(image_dir)
    labels = [label_dir / f"{image.stem}.txt" for image in images]
    return {
        "images": len(images),
        "labels": sum(1 for path in labels if path.exists()),
        "missing_labels": sum(1 for path in labels if not path.exists()),
    }
