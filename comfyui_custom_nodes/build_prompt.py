from __future__ import annotations

from collections import Counter

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _raw(text: str) -> str:
    return text


def _a(text: str) -> str:
    return "a " + text


def _picture(text: str) -> str:
    return "picture of a " + text


PROMPT_TEMPLATES = {"csl": _raw, "csl_a": _a, "csl_p": _picture}


class YOLOLabelToPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label_file": ("STRING", {"multiline": False, "default": "path/to/labels.txt"}),
                "data_yaml": ("STRING", {"multiline": False, "default": "path/to/data.yaml"}),
                "count_mode": (["repeat", "tag"], {"default": "tag"}),
                "repeat_instances": ("BOOLEAN", {"default": True}),
                "prompt_key": (["csl", "csl_a", "csl_p"], {"default": "csl_p"}),
                "fixed_prefix": ("STRING", {"multiline": False, "default": "stla9x"}),
                "fixed_suffix": ("STRING", {"multiline": False, "default": "realistic, real world"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "make_prompt"
    CATEGORY = "YOLO"

    def _load_class_names(self, yaml_path: str):
        if yaml is None:
            raise RuntimeError("PyYAML is required for YOLOLabelToPrompt.")
        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        names = data.get("names")
        if isinstance(names, list):
            return [str(item) for item in names]
        if isinstance(names, dict):
            keys = [int(key) for key in names]
            return [str(names.get(i, names.get(str(i), f"class_{i}"))) for i in range(max(keys) + 1)]
        raise ValueError("'names' in data.yaml must be a list or dict.")

    def _parse_labels(self, label_path: str):
        class_ids = []
        with open(label_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = raw_line.strip().split()
                if not parts:
                    continue
                try:
                    class_ids.append(int(float(parts[0])))
                except ValueError:
                    continue
        return class_ids

    @staticmethod
    def _class_name(class_id: int, class_names):
        return str(class_names[class_id]).strip() if 0 <= class_id < len(class_names) else f"class_{class_id}"

    def make_prompt(
        self,
        label_file,
        data_yaml,
        count_mode="tag",
        repeat_instances=True,
        prompt_key="csl_p",
        fixed_prefix="stla9x",
        fixed_suffix="realistic, real world",
    ):
        class_names = self._load_class_names(data_yaml)
        class_ids = self._parse_labels(label_file)
        pieces = []
        if fixed_prefix and fixed_prefix.strip():
            pieces.append(fixed_prefix.strip())
        if count_mode == "repeat":
            formatter = PROMPT_TEMPLATES.get(prompt_key, _raw)
            prompts = [formatter(self._class_name(class_id, class_names)) for class_id in class_ids]
            if not repeat_instances:
                prompts = list(dict.fromkeys(prompts))
            pieces.extend(prompts)
        else:
            counts = Counter(class_ids)
            for class_id in sorted(counts):
                pieces.append(f"{counts[class_id]} {self._class_name(class_id, class_names).lower()}")
        if fixed_suffix and fixed_suffix.strip():
            pieces.append(fixed_suffix.strip())
        return (", ".join(piece.strip() for piece in pieces if piece and piece.strip()),)


NODE_CLASS_MAPPINGS = {"YOLOLabelToPrompt": YOLOLabelToPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"YOLOLabelToPrompt": "YOLO Label to Prompt"}
