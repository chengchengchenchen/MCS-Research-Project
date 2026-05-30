from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from mcs_research.core.files import ensure_dir, materialize_file, write_json
from mcs_research.datasets.yolo import write_data_yaml

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def read_split_ids(voc_root: Path, split_name: str) -> list[str]:
    split_path = voc_root / "ImageSets" / "Main" / f"{split_name}.txt"
    return [line.strip().split()[0] for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def voc_box_to_yolo(box: tuple[float, float, float, float], width: float, height: float) -> str | None:
    xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(width, xmin))
    xmax = max(0.0, min(width, xmax))
    ymin = max(0.0, min(height, ymin))
    ymax = max(0.0, min(height, ymax))
    if xmax <= xmin or ymax <= ymin or width <= 0 or height <= 0:
        return None
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    return f"{x_center:.6f} {y_center:.6f} {(xmax - xmin) / width:.6f} {(ymax - ymin) / height:.6f}"


def parse_voc_annotation(annotation_path: Path, class_to_id: dict[str, int], include_difficult: bool) -> list[str]:
    root = ET.parse(annotation_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size block in {annotation_path}")
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            continue
        if not include_difficult and int(obj.findtext("difficult", "0") or 0) == 1:
            continue
        box_node = obj.find("bndbox")
        if box_node is None:
            continue
        box = (
            float(box_node.findtext("xmin", "0")),
            float(box_node.findtext("ymin", "0")),
            float(box_node.findtext("xmax", "0")),
            float(box_node.findtext("ymax", "0")),
        )
        yolo_box = voc_box_to_yolo(box, width, height)
        if yolo_box is not None:
            lines.append(f"{class_to_id[name]} {yolo_box}")
    return lines


def materialize_voc_split(
    *,
    devkit_root: Path,
    output_root: Path,
    output_split: str,
    source_sets: Sequence[Sequence[str]],
    class_to_id: dict[str, int],
    include_difficult: bool,
    image_storage: str,
) -> dict[str, Any]:
    image_out = output_root / output_split / "images"
    label_out = output_root / output_split / "labels"
    ensure_dir(image_out)
    ensure_dir(label_out)

    copied = labeled = annotations = 0
    for year, split_name in source_sets:
        voc_root = devkit_root / f"VOC{year}"
        for image_id in read_split_ids(voc_root, str(split_name)):
            source_image = voc_root / "JPEGImages" / f"{image_id}.jpg"
            source_annotation = voc_root / "Annotations" / f"{image_id}.xml"
            output_stem = f"VOC{year}_{image_id}"
            materialize_file(source_image, image_out / f"{output_stem}.jpg", image_storage)
            copied += 1
            lines = parse_voc_annotation(source_annotation, class_to_id, include_difficult)
            (label_out / f"{output_stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            annotations += len(lines)
            if lines:
                labeled += 1

    return {
        "split_name": output_split,
        "source_sets": [list(item) for item in source_sets],
        "num_images": copied,
        "num_labeled_images": labeled,
        "num_annotations": annotations,
    }


def convert_voc_to_yolo(
    *,
    devkit_root: Path,
    output_root: Path,
    train_sets: Sequence[Sequence[str]] = (("2007", "trainval"), ("2012", "trainval")),
    valid_sets: Sequence[Sequence[str]] = (("2007", "test"),),
    test_sets: Sequence[Sequence[str]] = (("2007", "test"),),
    include_difficult: bool = True,
    image_storage: str = "copy",
) -> dict[str, Any]:
    class_names = list(VOC_CLASSES)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    splits: dict[str, Any] = {}
    split_specs = {"train": train_sets, "valid": valid_sets, "test": test_sets}
    for split_name, source_sets in split_specs.items():
        if not source_sets:
            continue
        splits[split_name] = materialize_voc_split(
            devkit_root=devkit_root,
            output_root=output_root,
            output_split=split_name,
            source_sets=source_sets,
            class_to_id=class_to_id,
            include_difficult=include_difficult,
            image_storage=image_storage,
        )

    write_data_yaml(
        output_root / "data.yaml",
        class_names,
        train="train/images",
        val="valid/images" if "valid" in splits else "train/images",
        test="test/images" if "test" in splits else None,
    )
    manifest = {
        "format": "voc_to_yolo",
        "devkit_root": str(devkit_root),
        "output_root": str(output_root),
        "include_difficult": include_difficult,
        "class_names": class_names,
        "splits": splits,
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    return manifest
