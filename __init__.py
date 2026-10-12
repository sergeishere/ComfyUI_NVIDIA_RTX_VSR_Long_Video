"""Register NVIDIA RTX VSR Long Video with ComfyUI."""

from .nodes import RTX_VSR_Long_Video

NODE_CLASS_MAPPINGS = {"RTX_VSR_Long_Video": RTX_VSR_Long_Video}
NODE_DISPLAY_NAME_MAPPINGS = {"RTX_VSR_Long_Video": "RTX VSR Long Video"}
WEB_DIRECTORY = "./web"
