"""Bounded-memory, disk-backed FFmpeg pipeline for NVIDIA RTX VSR.

No ComfyUI or NVIDIA imports in this module: orchestration can be tested on CPU.
Input and output video are CFR RGB8/SDR. Only a chunk of frames is retained.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, ContextManager, Optional, Iterable

import imageio_ffmpeg

LOG = logging.getLogger("RTXVSRLongVideo")
VERSION = "0.1.11"
QUALITY_NAMES = (
    "BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA",
    "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA",
    "STREAMING_MEDIUM", "STREAMING_ULTRA",
    "DENOISE_LOW", "DENOISE_MEDIUM", "DENOISE_HIGH", "DENOISE_ULTRA",
    "DEBLUR_LOW", "DEBLUR_MEDIUM", "DEBLUR_HIGH", "DEBLUR_ULTRA",
)
PREPROCESS_NAMES = ("none",) + tuple(
    name for name in QUALITY_NAMES if name.startswith(("DENOISE_", "DEBLUR_"))
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts"}


@dataclass(frozen=True)
class Settings:
    scale: float = 2.0
    quality: str = "HIGHBITRATE_HIGH"
    strength: float = 1.0
    preprocess: str = "none"
    preprocess_strength: float = 1.0
    frames_per_chunk: int = 48
    encoder: str = "h264_nvenc"
    ffmpeg_path: str = ""
    fps_multiplier: int = 1

    def validate(self):
        if not (1.0 <= self.scale <= 4.0):
            raise ValueError("Scale must be between 1 and 4.")
        if self.quality not in QUALITY_NAMES:
            raise ValueError(f"Unknown RTX VSR quality: {self.quality}")
        if self.preprocess not in PREPROCESS_NAMES:
            raise ValueError(f"Unknown preprocessor: {self.preprocess}")
        if not (0.0 <= self.strength <= 1.0 and 0.0 <= self.preprocess_strength <= 1.0):
            raise ValueError("Strength must be between 0 and 1.")
        if not (4 <= self.frames_per_chunk <= 240):
            raise ValueError("frames_per_chunk must be between 4 and 240.")
        if self.encoder not in ("h264_nvenc", "hevc_nvenc", "libx264"):
            raise ValueError("Unsupported encoder.")
        if not (1 <= self.fps_multiplier <= 4):
            raise ValueError("fps_multiplier must be between 1 and 4.")


def output_dimensions(width: int, height: int, scale: float, quality: str) -> tuple[int, int]:
    if quality.startswith(("DEBLUR_", "DENOISE_")):
        return width, height
    # NVIDIA SDK expects an 8-pixel aligned output for upscale modes.
    w = max(8, round(width * scale / 8) * 8)
    h = max(8, round(height * scale / 8) * 8)
    if max(w, h) > 8192:
        raise ValueError(f"Requested output {w}x{h} exceeds the 8192-pixel video limit.")
    return w, h


def validate_encoder_dimensions(width: int, height: int, encoder: str) -> None:
    """Reject unsupported H.264 NVENC output before processing the first frame.

    NVIDIA's hardware H.264 encoder is limited to 4096 pixels per axis;
    HEVC NVENC on Ada can handle larger resolutions (up to 8192).
    """
    if encoder == "h264_nvenc" and (width > 4096 or height > 4096):
        raise ValueError(
            f"Output {width}x{height} exceeds h264_nvenc's 4096x4096 hardware limit. "
            "Choose hevc_nvenc for high-resolution NVIDIA encoding, "
            "or explicitly select libx264 for CPU H.264 encoding."
        )


@lru_cache(maxsize=12)
def ffmpeg_binary(encoder: str = "libx264", configured: str = "") -> str:
    """Resolve encoder-capable FFmpeg, never silently change the selected codec.

    imageio-ffmpeg bundled binaries frequently lack h264_nvenc. Prefer an
    NVENC-enabled system FFmpeg when hardware encoding was explicitly selected.
    """
    if configured.strip():
        candidates = [configured.strip().strip('"')]
    else:
        candidates = [shutil.which("ffmpeg"), imageio_ffmpeg.get_ffmpeg_exe()]
    for candidate in dict.fromkeys(filter(None, candidates)):
        if not Path(candidate).is_file():
            continue
        result = subprocess.run([candidate, "-hide_banner", "-encoders"],
                                capture_output=True, text=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode == 0 and re.search(r"\b" + re.escape(encoder) + r"\b", result.stdout):
            return str(candidate)
    raise RuntimeError(
        f"No FFmpeg executable supporting '{encoder}' was found. "
        "Install an NVENC-enabled FFmpeg and add it to PATH, enter its full "
        "path in ffmpeg_path, or manually select libx264."
    )


def fps_fraction(value: float) -> Fraction:
    if not value or value <= 0 or value > 480:
        raise ValueError(f"Invalid source frame rate: {value}")
    # imageio-ffmpeg rounds metadata FPS to two decimals; recover NTSC rates.
    for fps in (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(60000, 1001),
                Fraction(120000, 1001), Fraction(24, 1), Fraction(25, 1),
                Fraction(30, 1), Fraction(48, 1), Fraction(50, 1),
                Fraction(60, 1), Fraction(120, 1)):
        if abs(float(fps) - value) < 0.015:
            return fps
    return Fraction(str(value)).limit_denominator(100000)


def ffmpeg_rate(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


def _run_cmd(command: list[str]):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed ({result.returncode}): {result.stderr.decode('utf-8', 'replace')[-3000:]}")
    return result


def _encoder_args(encoder: str) -> list[str]:
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        return ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]


def _encode_chunk(path: Path, frame_data: Iterable[bytes], width: int, height: int,
                  fps: Fraction, encoder: str, ffmpeg_exe: str):
    """Stream frames to FFmpeg, and surface its stderr on pipe failures.

    Redirect stderr to a temporary file rather than a PIPE: this prevents
    producer/encoder deadlocks while very large frames are being written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.mkv")
    partial.unlink(missing_ok=True)
    args = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", ffmpeg_rate(fps), "-i", "pipe:0", "-an",
            *_encoder_args(encoder), "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-f", "matroska", str(partial)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with tempfile.TemporaryFile(mode="w+b") as ffmpeg_log:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=ffmpeg_log, creationflags=flags)
            try:
                assert proc.stdin is not None
                try:
                    for data in frame_data:
                        if len(data) != width * height * 3:
                            raise ValueError("RTX VSR produced an incorrectly sized RGB frame.")
                        proc.stdin.write(data)
                    proc.stdin.close()
                except OSError as exc:
                    # On Windows a closed FFmpeg stdin can raise EINVAL (22),
                    # rather than BrokenPipeError. Report the encoder's error.
                    if proc.stdin and not proc.stdin.closed:
                        try:
                            proc.stdin.close()
                        except OSError:
                            pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    ffmpeg_log.flush()
                    ffmpeg_log.seek(0)
                    details = ffmpeg_log.read().decode("utf-8", "replace")[-3000:].strip()
                    raise RuntimeError(
                        f"FFmpeg stopped accepting {width}x{height} RGB frames "
                        f"for {encoder}: {details or str(exc)}"
                    ) from exc
                status = proc.wait()
                if status:
                    ffmpeg_log.flush()
                    ffmpeg_log.seek(0)
                    details = ffmpeg_log.read().decode("utf-8", "replace")[-3000:]
                    raise RuntimeError(f"FFmpeg segment encoding failed ({encoder}, "
                                       f"{width}x{height}): {details}")
                if not partial.is_file() or partial.stat().st_size <= 0:
                    raise RuntimeError("FFmpeg did not write the segment.")
                os.replace(partial, path)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if proc.stdin and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
    finally:
        partial.unlink(missing_ok=True)



@contextmanager
def _work_dir(output_root: Path):
    """Create a disposable work directory that inherits the output folder's ACL.

    Not tempfile.TemporaryDirectory: on Windows (Python 3.12.4+) mkdtemp's 0o700
    applies an owner-only ACL, which os.replace carries onto the final MP4 and
    makes it unreadable over SMB.
    """
    path = output_root / f"_rtx_vsr_work_{uuid.uuid4().hex[:16]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _active_work_dir(path: Path):
    """Mark an executing run with an OS lock, released on process termination.

    Other ComfyUI instances can detect an active run and must not delete it.
    The lock is released before the work directory is deleted on Windows.
    """
    marker = path / ".active.lock"
    with marker.open("w+b") as handle:
        handle.write(b"1")
        handle.flush()
        _lock_work_file(handle, blocking=True)
        try:
            yield
        finally:
            _unlock_work_file(handle)


def _lock_work_file(handle, *, blocking: bool) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def _unlock_work_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cleanup_abandoned_work(output_root: Path) -> int:
    """Purge previous aborted runs, without touching finished output MP4 files.

    Only the node-owned temporary names are considered. Locked directories are
    active in another worker/process and are always preserved. The old v0.1.5
    'cache' tree is obsolete now that resume has been removed.
    """
    removed = 0
    for candidate in output_root.glob("_rtx_vsr_work_*"):
        if (candidate.is_symlink() or not candidate.is_dir() or
                not re.fullmatch(r"_rtx_vsr_work_[a-zA-Z0-9_-]{8,32}", candidate.name)):
            continue
        marker = candidate / ".active.lock"
        if marker.is_file():
            try:
                with marker.open("r+b") as handle:
                    try:
                        _lock_work_file(handle, blocking=False)
                    except OSError:
                        LOG.debug("RTX VSR: preserving active work directory %s", candidate)
                        continue
                    else:
                        _unlock_work_file(handle)
            except OSError:
                LOG.warning("RTX VSR: cannot inspect work directory %s; leaving intact", candidate)
                continue
        try:
            shutil.rmtree(candidate)
            removed += 1
        except OSError as exc:
            LOG.warning("RTX VSR: cannot remove abandoned work directory %s: %s", candidate, exc)

    old_cache = output_root / "cache"
    if old_cache.is_dir() and not old_cache.is_symlink():
        try:
            shutil.rmtree(old_cache)
            removed += 1
        except OSError as exc:
            LOG.warning("RTX VSR: cannot remove obsolete legacy cache %s: %s", old_cache, exc)

    # Before v0.1.7, a forced stop during final muxing could also leave a
    # temporary *.partial.mp4 alongside finished videos. Current versions
    # create their partial inside the locked work directory instead.
    for candidate in output_root.glob("*_RTXVSR_*.partial.mp4"):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if not re.fullmatch(r".+_RTXVSR_[0-9a-f]{12}\.partial\.mp4", candidate.name):
            continue
        try:
            candidate.unlink()
            removed += 1
        except OSError as exc:
            LOG.warning("RTX VSR: cannot remove obsolete partial output %s: %s", candidate, exc)

    if removed:
        LOG.info("RTX VSR: deleted %d abandoned temporary items before new run", removed)
    return removed

def _concat_list(cache_dir: Path, paths: list[Path]) -> Path:
    list_path = cache_dir / "concat.txt"
    # FFmpeg concat demuxer accepts slash-delimited paths, escaped for quotes.
    lines = []
    for path in paths:
        absolute = path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{absolute}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def _mux_output(cache_dir: Path, segments: list[Path], source: Path, destination: Path,
                ffmpeg_exe: str, encoder: str = "libx264"):
    playlist = _concat_list(cache_dir, segments)
    partial = cache_dir / "final.partial.mp4"
    partial.unlink(missing_ok=True)
    # Apple players (QuickTime, Safari, Photos) only accept HEVC in MP4 as 'hvc1';
    # FFmpeg's default 'hev1' tag makes the file unplayable on macOS/iOS.
    tag = ["-tag:v", "hvc1"] if encoder == "hevc_nvenc" else []
    common = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
              "-f", "concat", "-safe", "0", "-i", str(playlist), "-i", str(source),
              "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
              "-c:v", "copy", *tag, "-movflags", "+faststart"]
    try:
        try:
            _run_cmd([*common, "-c:a", "copy", str(partial)])
        except RuntimeError:
            # Some MP4 files don't support the original audio codec.
            LOG.warning("Audio stream-copy was rejected; retrying AAC for MP4 compatibility.")
            _run_cmd([*common, "-c:a", "aac", "-b:a", "192k", str(partial)])
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise RuntimeError("The final MP4 is empty.")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def process_video(source: Path, output_root: Path, settings: Settings,
                  transform_factory: Optional[Callable[..., ContextManager]] = None) -> dict:
    """Process a video frame-by-frame with disposable on-disk segments.

    Every execution recomputes the whole video. The work directory is removed
    after success or a handled failure/interruption; there is no resume cache.
    The injectable transform_factory allows FFmpeg/CPU regression testing.
    """
    from contextlib import ExitStack

    settings.validate()
    ffmpeg_exe = ffmpeg_binary(settings.encoder, settings.ffmpeg_path)
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Not a supported local video file: {source}")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _cleanup_abandoned_work(output_root)
    reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24")
    try:
        meta = next(reader)
        width, height = map(int, meta["size"])
        fps = fps_fraction(float(meta["fps"]))
        dst_width, dst_height = output_dimensions(width, height, settings.scale, settings.quality)
        validate_encoder_dimensions(dst_width, dst_height, settings.encoder)
        if width < 2 or height < 2:
            raise ValueError("Invalid video dimensions.")
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", source.stem)[:72]
        destination = output_root / f"{safe_stem}_RTXVSR_{uuid.uuid4().hex[:12]}.mp4"
        out_fps = fps * settings.fps_multiplier
        LOG.info("RTX VSR: %s → %s (%sx, %s, %s → %s FPS), %d frames/segment",
                 f"{width}x{height}", f"{dst_width}x{dst_height}", settings.scale,
                 settings.quality, ffmpeg_rate(fps), ffmpeg_rate(out_fps),
                 settings.frames_per_chunk)
        start = time.monotonic()
        total = 0
        segment_paths: list[Path] = []
        if transform_factory is None:
            from .vfx import video_transform as transform_factory

        # The work directory is also cleaned after exceptions and ComfyUI
        # interrupts (unlike a persistent resume cache).
        with _work_dir(output_root) as work, _active_work_dir(work):
            work_dir = Path(work)
            with ExitStack() as stack:
                transformer = None

                def outputs(frames: list[bytes], last_chunk: bool):
                    for i, data in enumerate(frames):
                        produced = transformer(data, last_chunk and i == len(frames) - 1)
                        yield from ([produced] if isinstance(produced, bytes) else produced)

                def finish_chunk(frames: list[bytes], last_chunk: bool = False):
                    nonlocal transformer
                    if transformer is None:
                        transformer = stack.enter_context(
                            transform_factory(width, height, dst_width, dst_height, settings)
                        )
                    path = work_dir / f"segment_{len(segment_paths):07d}.mkv"
                    _encode_chunk(path, outputs(frames, last_chunk),
                                  dst_width, dst_height, out_fps, settings.encoder, ffmpeg_exe)
                    segment_paths.append(path)
                    LOG.info("RTX VSR: encoded segment %d (%d frames)", len(segment_paths), len(frames))

                # Hold one chunk back so the final source frame is known: frame
                # generation pads the end of the video to keep its duration.
                frames: list[bytes] = []
                for raw in reader:
                    if len(frames) >= settings.frames_per_chunk:
                        finish_chunk(frames)
                        frames = []
                    frames.append(raw)
                    total += 1
                if frames:
                    finish_chunk(frames, last_chunk=True)
                    frames.clear()

            if not segment_paths:
                raise RuntimeError("The source contains no decodable video frames.")
            _mux_output(work_dir, segment_paths, source, destination, ffmpeg_exe, settings.encoder)

        duration = time.monotonic() - start
        result = {"output": str(destination), "frames": total,
                  "processed": total, "chunks": len(segment_paths),
                  "seconds": duration, "fps": ffmpeg_rate(out_fps),
                  "resolution": f"{dst_width}x{dst_height}"}
        LOG.info("RTX VSR complete: %d frames, %d segments, %.1fs → %s",
                 total, len(segment_paths), duration, destination)
        return result
    finally:
        reader.close()
