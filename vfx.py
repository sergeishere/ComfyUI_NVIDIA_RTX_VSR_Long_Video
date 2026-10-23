"""NVIDIA RTX VSR frame transformer. Intentionally no batch accumulation."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import logging
import numpy as np

LOG = logging.getLogger("RTXVSRLongVideo")

# nvidia-vfx 0.2.0.0 on driver r617: destroying an effect (close() or the
# native destructor) blocks forever inside NvTelemetryAPI64.dll while holding
# the GIL, freezing all of ComfyUI. Effects are therefore created once per
# configuration, reused across runs and never destroyed.
_EFFECTS: dict[tuple, object] = {}


@contextmanager
def video_transform(width, height, dst_width, dst_height, settings):
    import torch
    from nvvfx import VideoSuperRes

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA RTX VSR requires a CUDA-enabled NVIDIA RTX GPU.")
    device = torch.device("cuda", torch.cuda.current_device())
    gpu_index = device.index
    effect_kwargs = {"device": gpu_index, "image_encoding": VideoSuperRes.ImageEncoding.RGB8}

    def make_effect(quality, strength, out_w, out_h):
        key = (gpu_index, quality, float(strength), width, height, out_w, out_h)
        effect = _EFFECTS.get(key)
        if effect is not None:
            return effect
        effect = VideoSuperRes(quality=VideoSuperRes.QualityLevel[quality],
                               strength=float(strength), **effect_kwargs)
        effect.input_width = width
        effect.input_height = height
        effect.output_width = out_w
        effect.output_height = out_h
        effect.load()
        # Extra reference: never reach the hanging destructor, even at shutdown.
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(effect))
        _EFFECTS[key] = effect
        LOG.info("RTX VSR: cached %s effect %dx%d → %dx%d (%d cached, kept in VRAM)",
                 quality, width, height, out_w, out_h, len(_EFFECTS))
        return effect

    def make_frame_generation(multiplier):
        from nvvfx import VideoFrameGeneration
        from nvvfx.effects.video_frame_generation import Mode

        key = (gpu_index, "FRAME_GENERATION", multiplier, width, height)
        effect = _EFFECTS.get(key)
        if effect is not None:
            return effect
        effect = VideoFrameGeneration(mode=Mode.HIGH, **effect_kwargs)
        effect.input_width = width
        effect.input_height = height
        effect.frame_multiplier = multiplier
        effect.load()
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(effect))
        _EFFECTS[key] = effect
        LOG.info("RTX VSR: cached frame generation x%d effect %dx%d (%d cached, kept in VRAM)",
                 multiplier, width, height, len(_EFFECTS))
        return effect

    pre = None
    if settings.preprocess != "none":
        pre = make_effect(settings.preprocess, settings.preprocess_strength, width, height)
    sr = make_effect(settings.quality, settings.strength, dst_width, dst_height)
    multiplier = settings.fps_multiplier
    fg = make_frame_generation(multiplier) if multiplier > 1 else None
    previous = None  # last source frame on GPU, carried across segments

    def apply(rgb_bytes: bytes, is_last: bool = False) -> list[bytes]:
        """Return the output frames for one source frame.

        With frame generation, the interpolated frames between the previous and
        this source frame come first. The final source frame is repeated so the
        output keeps the source duration (and audio sync).
        """
        nonlocal previous
        # Copy bytes into writable ndarray; never retain source frames.
        image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
        gpu = torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32)
        gpu = gpu.permute(2, 0, 1).contiguous().div_(255.0)
        stream = torch.cuda.current_stream(device)
        ptr = int(stream.cuda_stream)
        if fg is None:
            return [enhance(gpu, stream, ptr)]
        outputs = []
        if previous is not None:
            for index in range(1, multiplier):
                middle = torch.from_dlpack(
                    fg.run(previous, gpu, frame_index=index, stream_ptr=ptr).image).clone()
                stream.synchronize()
                outputs.append(enhance(middle, stream, ptr))
        current = enhance(gpu, stream, ptr)
        outputs.append(current)
        if is_last:
            outputs.extend([current] * (multiplier - 1))
            previous = None
        else:
            previous = gpu
        return outputs

    def enhance(gpu, stream, ptr) -> bytes:
        if pre is not None:
            # SDK output capsule points to SDK-owned memory. Clone before reusing.
            pre_out = torch.from_dlpack(pre.run(gpu, stream_ptr=ptr).image).clone()
            stream.synchronize()
            del gpu
            gpu = pre_out
        enhanced = torch.from_dlpack(sr.run(gpu, stream_ptr=ptr).image).clone()
        stream.synchronize()
        cpu = (enhanced.clamp(0, 1).mul(255.0).round().to(torch.uint8)
               .permute(1, 2, 0).contiguous().cpu())
        result = cpu.numpy().tobytes()
        del cpu, enhanced
        return result

    yield apply
