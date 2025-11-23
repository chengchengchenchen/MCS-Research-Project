import os
import yaml

# ====== Prompt Engineering templates ======
RAW_PROMPT = lambda prompt: prompt
A_PROMPT = lambda prompt: "a " + prompt
P_PROMPT = lambda prompt: "picture of a " + prompt

PROMPT_TEMPLATES = {
    "csl": RAW_PROMPT,   # cat
    "csl_a": A_PROMPT,   # a cat
    "csl_p": P_PROMPT,   # picture of a cat
}


class YOLOLabelToPrompt:
    """
    Generate a prompt from YOLO label file and data.yaml.
    Construct text using PROMPT_TEMPLATES, for example:
    - csl     -> "cat, cat, dog"
    - csl_a   -> "a cat, a cat, a dog"
    - csl_p   -> "picture of a cat, picture of a cat, picture of a dog"

    Now also supports a fixed_suffix appended at the end, e.g.:
    "picture of a cat, picture of a dog, realistic, real world"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label_file": ("STRING", {
                    "multiline": False,
                    "default": "path/to/labels.txt",
                }),
                "data_yaml": ("STRING", {
                    "multiline": False,
                    "default": "path/to/data.yaml",
                }),
                # True: repeat classes by instance count; False: deduplicate
                "repeat_instances": ("BOOLEAN", {
                    "default": True,
                }),
                # Select prompt template (dropdown options)
                "prompt_key": (["csl", "csl_a", "csl_p"], {
                    "default": "csl_p",
                }),
                # Fixed prompt appended at the end
                "fixed_suffix": ("STRING", {
                    "multiline": False,
                    "default": "realistic, real world",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    FUNCTION = "make_prompt"
    CATEGORY = "YOLO"

    # ====== Parse data.yaml ======
    def _load_class_names(self, yaml_path):
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names")
        if names is None:
            raise ValueError("'names' field not found in data.yaml")

        # names: ['person', 'car', ...]
        if isinstance(names, list):
            return [str(n) for n in names]

        # names: {0: 'person', 1: 'car', ...}
        if isinstance(names, dict):
            max_idx = max(map(int, names.keys()))
            return [
                str(names.get(str(i), f"class_{i}"))
                for i in range(max_idx + 1)
            ]

        raise TypeError("'names' in data.yaml must be list or dict")

    # ====== Parse YOLO label file ======
    def _parse_labels(self, label_path):
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        class_ids = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cid = int(float(parts[0]))
                    class_ids.append(cid)
                except ValueError:
                    # Skip invalid lines
                    continue
        return class_ids

    # ====== Main logic: build prompt ======
    def make_prompt(self, label_file, data_yaml,
                    repeat_instances=True, prompt_key="csl_p",
                    fixed_suffix="realistic, real world"):

        class_names = self._load_class_names(data_yaml)
        class_ids = self._parse_labels(label_file)

        # 如果没有检测到任何目标：返回 fixed_suffix 或空字符串
        if not class_ids:
            fixed_suffix = fixed_suffix.strip()
            return (fixed_suffix if fixed_suffix else "",)

        # Get prompt template function; fallback to RAW_PROMPT
        pe = PROMPT_TEMPLATES.get(prompt_key, RAW_PROMPT)

        prompt_pieces = []
        for cid in class_ids:
            if 0 <= cid < len(class_names):
                base_name = str(class_names[cid])
            else:
                base_name = f"class_{cid}"
            prompt_pieces.append(pe(base_name))

        # Deduplicate prompts if needed
        if not repeat_instances:
            seen = set()
            deduped = []
            for p in prompt_pieces:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            prompt_pieces = deduped

        prompt = ", ".join(prompt_pieces)

        # ====== Append fixed_suffix at the end ======
        fixed_suffix = fixed_suffix.strip()
        if fixed_suffix:
            if prompt:
                prompt = f"{prompt}, {fixed_suffix}"
            else:
                prompt = fixed_suffix

        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "YOLOLabelToPrompt": YOLOLabelToPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOLabelToPrompt": "YOLO Label → Prompt"
}