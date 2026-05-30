from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from mcs_research.datasets.yolo import load_class_names, yolo_label_counts, format_label_counts

DEFAULT_STYLE_TOKEN = "stla9x"
DEFAULT_PROMPT = (
    "Return exactly 1 line of comma-separated lowercase tags for sd3.5 lora training. "
    "No <think> blocks, quotes, sentences, periods, or commentary. "
    "Order is strict: counts, materials/textures, background/setting, lighting, camera/framing. "
    "Use only commas as separators; no other delimiters. No leading/trailing whitespace. "
    "Counts section MUST be exactly: {label_counts}. Do not add or change count tags. "
    "For non-count groups: include at most 5 tags per group; omit uncertain tags. "
    "If a non-count group would be empty, output exactly one placeholder tag for that group: "
    "unknown_material, unknown_background, unknown_lighting, unknown_camera."
)


def request_json(url: str, data: bytes | None, timeout: int) -> dict:
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        message = f"HTTP {exc.code} {exc.reason}"
        if body:
            message = f"{message}: {body}"
        raise RuntimeError(message) from None


def build_prompt(base_prompt: str, label_counts: str) -> str:
    return base_prompt.format(label_counts=label_counts or "none")


def ensure_model_available(base_url: str, model: str, timeout: int) -> None:
    payload = request_json(f"{base_url.rstrip('/')}/api/tags", None, timeout)
    models = {item.get("name") for item in payload.get("models", []) if item.get("name")}
    if model not in models:
        available = ", ".join(sorted(models)) if models else "none"
        raise RuntimeError(f"Model '{model}' not found in /api/tags. Available: {available}")


def generate_caption(
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_path: Path,
    timeout: int,
    temperature: float,
) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }
    response = request_json(
        f"{base_url.rstrip('/')}/api/generate",
        json.dumps(payload).encode("utf-8"),
        timeout,
    )
    return str(response.get("response", "")).strip()


def caption_image(
    *,
    image_path: Path,
    label_path: Path,
    data_yaml: Path,
    model: str,
    base_url: str,
    style_token: str = DEFAULT_STYLE_TOKEN,
    prompt_template: str = DEFAULT_PROMPT,
    timeout: int = 180,
    temperature: float = 0.3,
) -> str:
    class_names = load_class_names(data_yaml)
    counts = Counter(yolo_label_counts(label_path))
    prompt = build_prompt(prompt_template, format_label_counts(counts, class_names))
    caption = generate_caption(
        base_url=base_url,
        model=model,
        prompt=prompt,
        image_path=image_path,
        timeout=timeout,
        temperature=temperature,
    )
    if not caption:
        raise RuntimeError(f"Empty caption for {image_path}")
    caption_text = f"{style_token}, {caption}" if style_token else caption
    image_path.with_suffix(".txt").write_text(caption_text + "\n", encoding="utf-8")
    return caption_text
