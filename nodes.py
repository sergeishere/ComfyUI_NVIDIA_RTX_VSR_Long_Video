"""ComfyUI node definition for the disk-backed NVIDIA RTX VSR pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import folder_paths

from .pipeline import Settings, QUALITY_NAMES, PREPROCESS_NAMES, process_video
from .video_input import connected_video_file
from .video_output import video_from_path

LOG = logging.getLogger("RTXVSRLongVideo")


class RTX_VSR_Long_Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO", {
                "tooltip": "Connect the VIDEO output from Load Video (or another native VIDEO node)."
            }),
            "scale": ("FLOAT", {
                "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                "tooltip": "Output size multiplier (1.0–4.0) for VSR upscale modes. "
                           "Ignored when quality is DEBLUR or DENOISE, which preserve the input resolution."
            }),
            "quality": (QUALITY_NAMES, {
                "default": "HIGHBITRATE_HIGH",
                "tooltip": "NVIDIA VFX effect to apply. HIGHBITRATE modes target clean video; "
                           "STREAMING modes enhance compressed sources; DEBLUR/DENOISE "
                           "restore the image without upscaling."
            }),
            "strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "NVIDIA effect intensity: 0 = minimum, 1 = maximum. "
                           "Applies to the selected quality mode."
            }),
            "preprocess": (PREPROCESS_NAMES, {
                "default": "none",
                "tooltip": "Optional extra first pass at the original resolution: "
                           "Deblur or Denoise before the main VSR pass. 'none' disables it."
            }),
            "preprocess_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Intensity of the optional Deblur/Denoise first pass. "
                           "Only used when preprocess is not 'none'."
            }),
            "frames_per_chunk": ("INT", {
                "default": 48, "min": 4, "max": 240, "step": 4,
                "tooltip": "Frames processed per temporary disk segment. Smaller values reduce "
                           "peak RAM but create more temporary files during processing."
            }),
            "encoder": (["h264_nvenc", "hevc_nvenc", "libx264"], {
                "tooltip": "Video encoder for temporary segments. H.264 NVENC is limited to 4096 pixels "
                           "per axis; choose HEVC NVENC for larger output (up to 8192 on RTX 40). "
                           "libx264 uses CPU. No automatic codec fallback."
            }),
        }}

    RETURN_TYPES = ("STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("video_path", "report", "video")
    FUNCTION = "upscale_video"
    CATEGORY = "NVIDIA/RTX VSR Long Video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Explicit Queue rerun must invoke node even when widget values are unchanged.
        return float("nan")

    def upscale_video(self, video, scale, quality, strength, preprocess,
                      preprocess_strength, frames_per_chunk, encoder):
        if video is None:
            raise ValueError("Connect a Load Video node to the 'video' input.")
        cfg = Settings(scale=scale, quality=quality, strength=strength,
                       preprocess=preprocess, preprocess_strength=preprocess_strength,
                       frames_per_chunk=frames_per_chunk, encoder=encoder)
        output_root = Path(folder_paths.get_output_directory()) / "NVIDIA_RTX_VSR_Long_Video"
        with connected_video_file(video, output_root) as connected_source:
            result = process_video(connected_source, output_root, cfg)
        summary = (f"{result['frames']} frames, {result['resolution']}, "
                   f"{result['processed']} processed, "
                   f"{result['chunks']} segments, {result['seconds']:.1f}s")
        LOG.info("RTX VSR: %s", summary)
        # Same native, file-backed VIDEO output as MiniMax H3 Final Decode.
        # No full-video IMAGE tensor, decode, or re-encode is performed.
        video_output = video_from_path(result["output"])
        filename = Path(result["output"]).name
        return {
            "ui": {
                "gifs": [{"filename": filename, "subfolder": "NVIDIA_RTX_VSR_Long_Video", "type": "output", "format": "video/mp4"}],
                "rtx_vsr_preview": [{"filename": filename, "subfolder": "NVIDIA_RTX_VSR_Long_Video", "type": "output", "format": "video/mp4"}],
                "rtx_vsr_report": [summary],
            },
            "result": (result["output"], summary, video_output),
        }

