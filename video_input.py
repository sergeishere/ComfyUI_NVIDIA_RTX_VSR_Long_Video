"""Resolve a ComfyUI VIDEO connection to a bounded-memory local file input.

For native Load Video / VideoFromFile, reuse the original file directly,
without copying it or calling get_components() (which materializes frames).
Other VIDEO types are exported to a temporary disk file using save_to().
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import logging
import uuid

import imageio_ffmpeg

LOG = logging.getLogger("RTXVSRLongVideo")


def _original_file_if_unchanged(video) -> Path | None:
    """Use native file-backed ComfyUI VIDEO only if no trim/crop is active.

    Importantly do not call get_stream_source() on arbitrary VIDEO objects:
    the VideoInput default builds a full in-memory BytesIO representation.
    """
    if type(video).__name__ != "VideoFromFile":
        return None
    if not callable(getattr(video, "get_stream_source", None)):
        return None
    source = video.get_stream_source()
    if not isinstance(source, (str, Path)):
        return None
    path = Path(source).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"The connected VIDEO source is not a file: {path}")

    trim = getattr(video, "get_active_trim_window", None)
    if callable(trim):
        start, duration = trim()
        if float(start) > 0 or float(duration) > 0:
            return None

    # A cropped VideoFromFile has the same original stream source, so compare
    # its advertised output dimensions against the actual encoded dimensions.
    dimensions = getattr(video, "get_dimensions", None)
    if callable(dimensions):
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            encoded = tuple(int(v) for v in metadata["size"])
        finally:
            reader.close()
        if tuple(dimensions()) != encoded:
            return None

    return path


@contextmanager
def connected_video_file(video, output_root: Path):
    """Yield a file suitable for the existing FFmpeg pipeline.

    Generated/trimmed/cropped VIDEO inputs need an on-disk staging file.
    The staging file is removed after processing or failure. The persistent
    resume cache remains available for native file-backed Load Video inputs.
    """
    if video is None:
        raise ValueError("Connected VIDEO must not be None.")
    direct = _original_file_if_unchanged(video)
    if direct is not None:
        LOG.info("RTX VSR: using connected VIDEO source directly: %s", direct)
        yield direct
        return

    save_to = getattr(video, "save_to", None)
    if not callable(save_to):
        raise ValueError("Connected VIDEO does not implement ComfyUI's save_to() API.")
    staging = Path(output_root) / "_video_input_staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged_path = staging / f"connected_{uuid.uuid4().hex}.mp4"
    LOG.info("RTX VSR: exporting connected VIDEO to temporary disk file: %s", staged_path)
    try:
        # Let ComfyUI apply trims/crops, where present, rather than quietly
        # processing the original untrimmed/unmodified file.
        save_to(str(staged_path))
        if not staged_path.is_file() or staged_path.stat().st_size == 0:
            raise RuntimeError("Connected VIDEO export created no usable file.")
        yield staged_path
    finally:
        staged_path.unlink(missing_ok=True)
        try:
            staging.rmdir()
        except OSError:
            pass
