"""Expose a completed MP4 as ComfyUI's native file-backed VIDEO type."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path


def video_from_path(path: str | Path):
    """Wrap the final on-disk MP4, without decoding or copying any frames."""
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"RTX VSR output is not a file: {source}")

    # The public location has changed between ComfyUI builds; match Extender.
    last_error = None
    for module_name in (
        "comfy_api.latest._input_impl.video_types",
        "comfy_api.latest.input_impl.video_types",
    ):
        try:
            video_type = getattr(import_module(module_name), "VideoFromFile")
        except (ImportError, AttributeError) as exc:
            last_error = exc
            continue
        return video_type(str(source))

    raise RuntimeError(
        "RTX VSR Long Video: native VIDEO output requires a ComfyUI build "
        f"with VideoFromFile support ({last_error})."
    )
