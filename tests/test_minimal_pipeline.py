from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mcs_research.core.files import write_json
from mcs_research.datasets.voc import convert_voc_to_yolo
from mcs_research.datasets.yolo import convert_coco_to_yolo
from mcs_research.filtering.synthetic import (
    FilterConfig,
    MetricStats,
    ScoreWeights,
    filter_yolo_dataset,
    select_candidates,
)
from mcs_research.training.yolo import build_train_kwargs, run_yolo_training


def test_coco_to_yolo_conversion(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (20, 10), "white").save(image_root / "sample.jpg")
    annotation_path = tmp_path / "ann.json"
    write_json(
        annotation_path,
        {
            "images": [{"id": 1, "file_name": "sample.jpg", "width": 20, "height": 10}],
            "categories": [{"id": 3, "name": "object"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [5, 2, 10, 4], "iscrowd": 0}],
        },
    )

    summary = convert_coco_to_yolo(
        annotation_path=annotation_path,
        image_root=image_root,
        output_root=tmp_path / "out",
        split_name="train",
    )

    assert summary["num_images"] == 1
    assert (tmp_path / "out" / "train" / "images" / "sample.jpg").exists()
    assert (tmp_path / "out" / "train" / "labels" / "sample.txt").read_text(encoding="utf-8").strip() == (
        "0 0.500000 0.400000 0.500000 0.400000"
    )


def test_voc_to_yolo_conversion(tmp_path: Path) -> None:
    voc_root = tmp_path / "VOCdevkit" / "VOC2007"
    for rel in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (voc_root / rel).mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(voc_root / "JPEGImages" / "000001.jpg")
    (voc_root / "ImageSets" / "Main" / "trainval.txt").write_text("000001\n", encoding="utf-8")
    (voc_root / "Annotations" / "000001.xml").write_text(
        """<annotation>
  <size><width>20</width><height>10</height><depth>3</depth></size>
  <object>
    <name>person</name><difficult>0</difficult>
    <bndbox><xmin>5</xmin><ymin>2</ymin><xmax>15</xmax><ymax>6</ymax></bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )

    summary = convert_voc_to_yolo(
        devkit_root=tmp_path / "VOCdevkit",
        output_root=tmp_path / "voc_out",
        train_sets=[["2007", "trainval"]],
        valid_sets=[],
        test_sets=[],
    )

    label_text = (tmp_path / "voc_out" / "train" / "labels" / "VOC2007_000001.txt").read_text(encoding="utf-8").strip()
    assert summary["splits"]["train"]["num_images"] == 1
    assert label_text == "14 0.500000 0.400000 0.500000 0.400000"


def test_filter_selection_matches_golden_fixture() -> None:
    items = [
        {"index": 0, "clip_score": 0.10, "dino_score": 0.40, "bboxes": [{"clip_score": 0.20}]},
        {"index": 1, "clip_score": 0.80, "dino_score": 0.90, "bboxes": [{"clip_score": 0.70}]},
        {"index": 2, "clip_score": 0.60, "dino_score": 0.50, "bboxes": [{"clip_score": 0.60}]},
    ]
    new_stats = {
        key: MetricStats(p5=0.1, p25=0.2, p50=0.5, p75=0.7, p95=0.9)
        for key in ("clip", "dino", "bbox_mean", "bbox_min")
    }
    new_cfg = FilterConfig(
        synth_keep_ratio=0.75,
        synth_keep_topk=-1,
        synth_min_keep=0,
        filter_quantile=25.0,
        weights=ScoreWeights(clip=0.2, dino=0.5, bbox_mean=0.25, bbox_min=0.05),
    )

    new_selected, new_dropped = select_candidates(items, new_stats, new_cfg, stem="sample", synth_dir=Path("."))

    assert [item.index for item in new_selected] == [1, 2]
    assert new_dropped == 1


def test_filter_dataset_tiny_fixture(tmp_path: Path) -> None:
    real_images = tmp_path / "source" / "train" / "original" / "images"
    real_labels = tmp_path / "source" / "train" / "original" / "labels"
    synth_root = tmp_path / "source" / "train" / "synth" / "sample"
    synth_labels = tmp_path / "source" / "train" / "synth_labels"
    for path in (real_images, real_labels, synth_root, synth_labels):
        path.mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(real_images / "sample.jpg")
    real_labels.joinpath("sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    synth_labels.joinpath("sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    Image.new("RGB", (10, 10), "gray").save(synth_root / "sample_00.png")
    Image.new("RGB", (10, 10), "black").save(synth_root / "sample_01.png")
    write_json(
        synth_root / "scores.json",
        {
            "items": [
                {"index": 0, "clip_score": 0.1, "dino_score": 0.1, "bboxes": [{"clip_score": 0.1}]},
                {"index": 1, "clip_score": 0.9, "dino_score": 0.9, "bboxes": [{"clip_score": 0.9}]},
            ]
        },
    )

    summary = filter_yolo_dataset(
        output_root=tmp_path / "out",
        splits={
            "train": {
                "real_image_dir": str(real_images),
                "real_label_dir": str(real_labels),
                "synth_root": str(synth_root.parent),
                "synth_label_dir": str(synth_labels),
            }
        },
        cfg=FilterConfig(synth_keep_ratio=0.5, synth_keep_topk=-1, filter_quantile=0.0),
        class_names=["object"],
    )

    assert summary["splits"]["train"]["originals"] == 1
    assert summary["splits"]["train"]["synthetic"] == 1
    assert (tmp_path / "out" / "train" / "images" / "sample_01.png").exists()
    assert (tmp_path / "out" / "data.yaml").exists()


def test_training_dry_run_payload(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images\nval: images\nnames: ['object']\n", encoding="utf-8")
    cfg = {
        "data": str(data_yaml),
        "weights": "yolov8s.pt",
        "project": str(tmp_path / "runs"),
        "name": "smoke",
        "seed": 3,
        "dry_run": True,
    }

    kwargs = build_train_kwargs(cfg)
    payload = run_yolo_training(cfg)

    assert kwargs["seed"] == 3
    assert payload["dry_run"] is True
    assert payload["train_kwargs"]["name"] == "smoke"
