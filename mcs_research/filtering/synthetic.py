from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageChops, ImageStat

from mcs_research.core.files import (
    clear_dir,
    ensure_dir,
    find_image,
    list_images,
    load_json,
    materialize_file,
    write_image_list,
    write_json,
)
from mcs_research.datasets.yolo import load_class_names, write_data_yaml


@dataclass
class MetricStats:
    p5: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0

    def normalize(self, value: float) -> float:
        if self.p95 <= self.p5:
            return 0.0
        return max(0.0, min(1.0, (value - self.p5) / (self.p95 - self.p5)))

    def floor(self, user_floor: float | None, quantile: float = 0.25) -> float:
        if user_floor is not None:
            return user_floor
        if quantile <= 0.05:
            return self.p5
        if quantile <= 0.25:
            return self.p25
        if quantile <= 0.5:
            return self.p50
        if quantile <= 0.75:
            return self.p75
        return self.p95


@dataclass
class ScoreWeights:
    clip: float = 0.2
    dino: float = 0.5
    bbox_mean: float = 0.25
    bbox_min: float = 0.05

    def total(self) -> float:
        return self.clip + self.dino + self.bbox_mean + self.bbox_min


@dataclass
class FilterConfig:
    include_real: bool = True
    include_synth: bool = True
    synth_keep_ratio: float = 0.3
    synth_keep_topk: int = -1
    synth_min_keep: int = 0
    filter_quantile: float = 25.0
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    clip_floor: float | None = None
    dino_floor: float | None = None
    bbox_mean_floor: float | None = None
    bbox_min_floor: float | None = None
    synth_max_channel_diff: float | None = None
    synth_exact_count: int = 0
    retention_mode: str = "global"
    retention_class_quotas: dict[str, int] = field(default_factory=dict)
    retention_class_floors: dict[str, int] = field(default_factory=dict)
    retention_allow_quota_rollover: bool = True
    retention_stem_cap: int = 0
    sample_total: int = 0
    sample_seed: int = 1
    image_storage: str = "copy"
    write_image_lists: bool = False


@dataclass(frozen=True)
class ItemQuality:
    index: int
    score: float
    clip_raw: float
    dino_raw: float
    bbox_mean: float
    bbox_min: float


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def build_metric_stats(values: Sequence[float]) -> MetricStats:
    return MetricStats(
        p5=percentile(values, 5),
        p25=percentile(values, 25),
        p50=percentile(values, 50),
        p75=percentile(values, 75),
        p95=percentile(values, 95),
    )


def candidate_bbox_scores(item: Mapping[str, Any]) -> tuple[float, float]:
    bboxes = item.get("bboxes") or []
    scores = [float(box.get("clip_score", 0.0)) for box in bboxes if isinstance(box, Mapping)]
    if not scores:
        return 0.0, 0.0
    return sum(scores) / len(scores), min(scores)


def collect_metric_stats(synth_root: Path) -> dict[str, MetricStats]:
    clip_vals: list[float] = []
    dino_vals: list[float] = []
    bbox_mean_vals: list[float] = []
    bbox_min_vals: list[float] = []
    for score_path in sorted(synth_root.glob("*/scores.json")):
        payload = load_json(score_path)
        for item in payload.get("items", []):
            clip_vals.append(float(item.get("clip_score", 0.0)))
            dino_vals.append(float(item.get("dino_score", 0.0)))
            bbox_mean, bbox_min = candidate_bbox_scores(item)
            if item.get("bboxes"):
                bbox_mean_vals.append(bbox_mean)
                bbox_min_vals.append(bbox_min)
    return {
        "clip": build_metric_stats(clip_vals),
        "dino": build_metric_stats(dino_vals),
        "bbox_mean": build_metric_stats(bbox_mean_vals),
        "bbox_min": build_metric_stats(bbox_min_vals),
    }


@lru_cache(maxsize=100_000)
def compute_max_mean_channel_diff(image_path: str) -> float:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        r, g, b = rgb.split()
        rg = float(ImageStat.Stat(ImageChops.difference(r, g)).mean[0])
        rb = float(ImageStat.Stat(ImageChops.difference(r, b)).mean[0])
        gb = float(ImageStat.Stat(ImageChops.difference(g, b)).mean[0])
    return max(rg, rb, gb)


def resolve_item_channel_diff(item: dict[str, Any], *, stem: str, synth_dir: Path) -> float | None:
    raw = item.get("max_mean_channel_diff", item.get("mean_channel_diff"))
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    index = int(item.get("index", 0))
    image_path = find_synth_image(synth_dir, stem, index)
    if image_path is None:
        return None
    value = compute_max_mean_channel_diff(str(image_path.resolve()))
    item["max_mean_channel_diff"] = value
    return value


def compute_quality(item: Mapping[str, Any], stats: Mapping[str, MetricStats], cfg: FilterConfig) -> ItemQuality | None:
    clip_raw = float(item.get("clip_score", 0.0))
    dino_raw = float(item.get("dino_score", 0.0))
    bbox_mean, bbox_min = candidate_bbox_scores(item)

    fq = cfg.filter_quantile / 100.0
    floors = {
        "clip": stats["clip"].floor(cfg.clip_floor, fq),
        "dino": stats["dino"].floor(cfg.dino_floor, fq),
        "bbox_mean": stats["bbox_mean"].floor(cfg.bbox_mean_floor, fq),
        "bbox_min": stats["bbox_min"].floor(cfg.bbox_min_floor, fq),
    }
    if clip_raw < floors["clip"] or dino_raw < floors["dino"] or bbox_mean < floors["bbox_mean"] or bbox_min < floors["bbox_min"]:
        return None

    denom = cfg.weights.total() or 1.0
    combined = (
        stats["clip"].normalize(clip_raw) * cfg.weights.clip
        + stats["dino"].normalize(dino_raw) * cfg.weights.dino
        + stats["bbox_mean"].normalize(bbox_mean) * cfg.weights.bbox_mean
        + stats["bbox_min"].normalize(bbox_min) * cfg.weights.bbox_min
    ) / denom
    return ItemQuality(
        index=int(item.get("index", 0)),
        score=float(combined),
        clip_raw=clip_raw,
        dino_raw=dino_raw,
        bbox_mean=bbox_mean,
        bbox_min=bbox_min,
    )


def choose_keep_count(num_candidates: int, cfg: FilterConfig) -> int:
    keep_n = math.ceil(num_candidates * max(0.0, min(1.0, cfg.synth_keep_ratio)))
    if cfg.synth_keep_topk >= 0:
        keep_n = min(keep_n, cfg.synth_keep_topk)
    keep_n = max(cfg.synth_min_keep, keep_n)
    return min(keep_n, num_candidates)


def select_candidates(
    items: Sequence[dict[str, Any]],
    stats: Mapping[str, MetricStats],
    cfg: FilterConfig,
    *,
    stem: str,
    synth_dir: Path,
) -> tuple[list[ItemQuality], int]:
    accepted: list[ItemQuality] = []
    dropped = 0
    for item in items:
        if cfg.synth_max_channel_diff is not None:
            channel_diff = resolve_item_channel_diff(item, stem=stem, synth_dir=synth_dir)
            if channel_diff is None or channel_diff > cfg.synth_max_channel_diff:
                dropped += 1
                continue
        quality = compute_quality(item, stats, cfg)
        if quality is None:
            dropped += 1
            continue
        accepted.append(quality)
    accepted.sort(key=lambda item: item.score, reverse=True)
    keep_n = choose_keep_count(len(accepted), cfg)
    return accepted[:keep_n], dropped + max(0, len(accepted) - keep_n)


def find_synth_image(synth_dir: Path, stem: str, index: int) -> Path | None:
    return find_image(synth_dir, f"{stem}_{index:02d}")


def primary_label_name(label_path: Path, class_names: Sequence[str] | None = None) -> str | None:
    counts: dict[int, int] = {}
    area_by_class: dict[int, float] = {}
    if not label_path.exists():
        return None
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        counts[class_id] = counts.get(class_id, 0) + 1
        area_by_class[class_id] = area_by_class.get(class_id, 0.0) + max(0.0, width * height)
    if not counts:
        return None
    class_id = sorted(counts, key=lambda cid: (-area_by_class.get(cid, 0.0), -counts[cid], cid))[0]
    if class_names is not None and 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return f"class_{class_id}"


def _pending_sort_key(item: Mapping[str, Any]) -> tuple[float, str, str]:
    return (-float(item["score"]), str(item.get("image_name") or ""), str(item.get("stem") or ""))


def finalize_exact_selection(pending: list[dict[str, Any]], cfg: FilterConfig) -> tuple[list[dict[str, Any]], int]:
    exact_target = max(0, int(cfg.synth_exact_count))
    if exact_target <= 0:
        return sorted(pending, key=_pending_sort_key), 0
    if cfg.retention_mode in {"class_balanced", "class_floor"}:
        quotas = cfg.retention_class_quotas if cfg.retention_mode == "class_balanced" else cfg.retention_class_floors
        selected = _select_class_quota(pending, exact_target, quotas, cfg.retention_allow_quota_rollover)
    elif cfg.retention_stem_cap > 0:
        selected = _select_with_stem_cap(pending, exact_target, cfg.retention_stem_cap)
    else:
        selected = sorted(pending, key=_pending_sort_key)[:exact_target]
    return selected[:exact_target], max(0, exact_target - len(selected))


def _select_class_quota(
    pending: list[dict[str, Any]],
    exact_target: int,
    quotas: Mapping[str, int],
    allow_rollover: bool,
) -> list[dict[str, Any]]:
    if not quotas:
        return sorted(pending, key=_pending_sort_key)[:exact_target]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in pending:
        grouped.setdefault(str(item.get("label_name") or ""), []).append(item)
    for items in grouped.values():
        items.sort(key=_pending_sort_key)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for class_name in sorted(quotas):
        for item in grouped.get(class_name, [])[: max(0, int(quotas[class_name]))]:
            selected.append(item)
            selected_ids.add(id(item))
    if len(selected) < exact_target and allow_rollover:
        for item in sorted(pending, key=_pending_sort_key):
            if id(item) in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= exact_target:
                break
    selected.sort(key=_pending_sort_key)
    return selected[:exact_target]


def _select_with_stem_cap(pending: list[dict[str, Any]], exact_target: int, stem_cap: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_stem: dict[str, int] = {}
    for item in sorted(pending, key=_pending_sort_key):
        stem = str(item.get("stem") or "")
        if by_stem.get(stem, 0) >= stem_cap:
            continue
        selected.append(item)
        by_stem[stem] = by_stem.get(stem, 0) + 1
        if len(selected) >= exact_target:
            break
    return selected


def materialize_filtered_split(
    *,
    real_image_dir: Path,
    real_label_dir: Path,
    output_root: Path,
    split: str,
    cfg: FilterConfig,
    stats: Mapping[str, MetricStats] | None = None,
    synth_root: Path | None = None,
    synth_label_dir: Path | None = None,
    keep_originals: int | None = None,
    rng: random.Random | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    out_image_dir = output_root / split / "images"
    out_label_dir = output_root / split / "labels"
    ensure_dir(out_image_dir)
    ensure_dir(out_label_dir)
    output_images: list[Path] = []

    originals = list_images(real_image_dir)
    if keep_originals is not None and keep_originals > 0 and keep_originals < len(originals):
        originals = (rng or random.Random(cfg.sample_seed)).sample(originals, keep_originals)
    elif keep_originals is not None and keep_originals <= 0:
        originals = []

    stats = stats or (collect_metric_stats(synth_root) if synth_root is not None else {})
    missing_label = missing_scores = missing_img = missing_synth_label = filtered = 0
    num_real = num_synth = 0
    pending_synth: list[dict[str, Any]] = []

    for image_path in originals:
        stem = image_path.stem
        label_path = real_label_dir / f"{stem}.txt"
        if not label_path.exists():
            missing_label += 1
            continue
        if cfg.include_real:
            output_image = out_image_dir / image_path.name
            materialize_file(image_path, output_image, cfg.image_storage)
            materialize_file(label_path, out_label_dir / label_path.name, "copy")
            output_images.append(output_image)
            num_real += 1
        if not cfg.include_synth or synth_root is None:
            continue
        synth_dir = synth_root / stem
        score_path = synth_dir / "scores.json"
        if not score_path.exists():
            missing_scores += 1
            continue
        payload = load_json(score_path)
        selected, dropped = select_candidates(payload.get("items", []), stats, cfg, stem=stem, synth_dir=synth_dir)
        filtered += dropped
        for quality in selected:
            synth_image = find_synth_image(synth_dir, stem, quality.index)
            if synth_image is None:
                missing_img += 1
                continue
            source_label = (synth_label_dir / f"{stem}.txt") if synth_label_dir is not None else label_path
            if not source_label.exists():
                missing_synth_label += 1
                continue
            label_name = f"{synth_image.stem}.txt"
            if cfg.synth_exact_count > 0:
                pending_synth.append(
                    {
                        "score": quality.score,
                        "image_name": synth_image.name,
                        "image_path": synth_image,
                        "label_path": source_label,
                        "label_name": primary_label_name(source_label, class_names),
                        "stem": stem,
                        "slbl_name": label_name,
                    }
                )
            else:
                output_image = out_image_dir / synth_image.name
                materialize_file(synth_image, output_image, cfg.image_storage)
                materialize_file(source_label, out_label_dir / label_name, "copy")
                output_images.append(output_image)
                num_synth += 1

    exact_shortfall = 0
    if cfg.include_synth and cfg.synth_exact_count > 0:
        kept_synth, exact_shortfall = finalize_exact_selection(pending_synth, cfg)
        filtered += max(0, len(pending_synth) - len(kept_synth))
        for item in kept_synth:
            output_image = out_image_dir / str(item["image_name"])
            materialize_file(Path(item["image_path"]), output_image, cfg.image_storage)
            materialize_file(Path(item["label_path"]), out_label_dir / str(item["slbl_name"]), "copy")
            output_images.append(output_image)
            num_synth += 1

    image_list_path = None
    if cfg.write_image_lists:
        image_list_path = output_root / f"{split}.txt"
        write_image_list(image_list_path, output_images)
    return {
        "originals": num_real,
        "synthetic": num_synth,
        "filtered": filtered,
        "missing_label": missing_label,
        "missing_scores": missing_scores,
        "missing_img": missing_img,
        "missing_synth_label": missing_synth_label,
        "synth_exact_count": max(0, int(cfg.synth_exact_count)),
        "synth_exact_shortfall": exact_shortfall,
        "image_storage": cfg.image_storage,
        "image_list_path": str(image_list_path.resolve()) if image_list_path is not None else None,
    }


def filter_yolo_dataset(
    *,
    output_root: Path,
    splits: Mapping[str, Mapping[str, Any]],
    cfg: FilterConfig,
    class_names: Sequence[str] | None = None,
    data_yaml: Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    if overwrite:
        clear_dir(output_root)
    else:
        ensure_dir(output_root)
    if class_names is None and data_yaml is not None:
        class_names = load_class_names(data_yaml)
    class_names = list(class_names or [])

    rng = random.Random(cfg.sample_seed)
    sample_counts = _sample_counts(list(splits.keys()), cfg.sample_total)
    summaries: dict[str, Any] = {}
    for split, raw_spec in splits.items():
        spec = dict(raw_spec)
        synth_root = Path(spec["synth_root"]).resolve() if spec.get("synth_root") else None
        stats = collect_metric_stats(synth_root) if synth_root is not None and synth_root.exists() else None
        summaries[split] = materialize_filtered_split(
            real_image_dir=Path(spec["real_image_dir"]).resolve(),
            real_label_dir=Path(spec["real_label_dir"]).resolve(),
            output_root=output_root,
            split=split,
            cfg=cfg,
            stats=stats,
            synth_root=synth_root,
            synth_label_dir=Path(spec["synth_label_dir"]).resolve() if spec.get("synth_label_dir") else None,
            keep_originals=sample_counts.get(split),
            rng=rng,
            class_names=class_names,
        )

    train_path = f"{list(splits.keys())[0]}/images"
    val_path = "valid/images" if "valid" in splits else train_path
    test_path = "test/images" if "test" in splits else None
    if class_names:
        write_data_yaml(output_root / "data.yaml", class_names, train=train_path, val=val_path, test=test_path)
    payload = {
        "output_root": str(output_root.resolve()),
        "splits": summaries,
        "config": filter_config_to_dict(cfg),
        "class_names": class_names,
    }
    write_json(output_root / "filter_summary.json", payload)
    return payload


def filter_config_from_mapping(payload: Mapping[str, Any]) -> FilterConfig:
    weights = payload.get("weights") or {}
    return FilterConfig(
        include_real=bool(payload.get("include_real", True)),
        include_synth=bool(payload.get("include_synth", True)),
        synth_keep_ratio=float(payload.get("synth_keep_ratio", 0.3)),
        synth_keep_topk=int(payload.get("synth_keep_topk", payload.get("synth_topk", -1))),
        synth_min_keep=int(payload.get("synth_min_keep", 0)),
        filter_quantile=float(payload.get("filter_quantile", 25.0)),
        weights=ScoreWeights(
            clip=float(payload.get("weight_clip", weights.get("clip", 0.2))),
            dino=float(payload.get("weight_dino", weights.get("dino", 0.5))),
            bbox_mean=float(payload.get("weight_bbox_mean", weights.get("bbox_mean", 0.25))),
            bbox_min=float(payload.get("weight_bbox_min", weights.get("bbox_min", 0.05))),
        ),
        clip_floor=_optional_float(payload.get("clip_floor")),
        dino_floor=_optional_float(payload.get("dino_floor")),
        bbox_mean_floor=_optional_float(payload.get("bbox_mean_floor")),
        bbox_min_floor=_optional_float(payload.get("bbox_min_floor")),
        synth_max_channel_diff=_optional_float(payload.get("synth_max_channel_diff")),
        synth_exact_count=int(payload.get("synth_exact_count", 0)),
        retention_mode=str(payload.get("retention_mode", "global")),
        retention_class_quotas={str(k): int(v) for k, v in dict(payload.get("retention_class_quotas", {})).items()},
        retention_class_floors={str(k): int(v) for k, v in dict(payload.get("retention_class_floors", {})).items()},
        retention_allow_quota_rollover=bool(payload.get("retention_allow_quota_rollover", True)),
        retention_stem_cap=int(payload.get("retention_stem_cap", 0)),
        sample_total=int(payload.get("sample_total", 0)),
        sample_seed=int(payload.get("sample_seed", payload.get("seed", 1))),
        image_storage=str(payload.get("image_storage", "copy")),
        write_image_lists=bool(payload.get("write_image_lists", payload.get("use_image_lists", False))),
    )


def filter_config_to_dict(cfg: FilterConfig) -> dict[str, Any]:
    return {
        "include_real": cfg.include_real,
        "include_synth": cfg.include_synth,
        "synth_keep_ratio": cfg.synth_keep_ratio,
        "synth_keep_topk": cfg.synth_keep_topk,
        "synth_min_keep": cfg.synth_min_keep,
        "filter_quantile": cfg.filter_quantile,
        "weights": {
            "clip": cfg.weights.clip,
            "dino": cfg.weights.dino,
            "bbox_mean": cfg.weights.bbox_mean,
            "bbox_min": cfg.weights.bbox_min,
        },
        "synth_exact_count": cfg.synth_exact_count,
        "retention_mode": cfg.retention_mode,
        "sample_seed": cfg.sample_seed,
        "image_storage": cfg.image_storage,
    }


def _sample_counts(splits: Sequence[str], sample_total: int) -> dict[str, int | None]:
    sample_counts: dict[str, int | None] = {split: None for split in splits}
    if sample_total <= 0:
        return sample_counts
    if "train" in sample_counts and "valid" in sample_counts:
        train_n = max(1, int(round(sample_total * 0.8)))
        valid_n = max(0, sample_total - train_n)
        if valid_n == 0:
            valid_n = 1
            train_n = max(0, sample_total - valid_n)
        sample_counts["train"] = train_n
        sample_counts["valid"] = valid_n
    else:
        per_split = max(1, int(math.ceil(sample_total / len(splits))))
        for split in splits:
            sample_counts[split] = per_split
    return sample_counts


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
