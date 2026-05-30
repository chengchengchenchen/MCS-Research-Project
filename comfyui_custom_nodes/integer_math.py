class IntegerMin:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {"default": 0}), "b": ("INT", {"default": 0})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "compute"
    CATEGORY = "math/int"

    def compute(self, a, b):
        return (min(int(a), int(b)),)


class IntegerMax:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {"default": 0}), "b": ("INT", {"default": 0})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "compute"
    CATEGORY = "math/int"

    def compute(self, a, b):
        return (max(int(a), int(b)),)


NODE_CLASS_MAPPINGS = {"IntegerMin": IntegerMin, "IntegerMax": IntegerMax}
NODE_DISPLAY_NAME_MAPPINGS = {"IntegerMin": "Integer Min", "IntegerMax": "Integer Max"}
