"""NVIDIA RTX VSR frame transformer. Intentionally no batch accumulation."""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np


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
        effect = VideoSuperRes(quality=VideoSuperRes.QualityLevel[quality],
                               strength=float(strength), **effect_kwargs)
        effect.input_width = width
        effect.input_height = height
        effect.output_width = out_w
        effect.output_height = out_h
        effect.load()
        return effect

    pre = None
    sr = None
    try:
        if settings.preprocess != "none":
            pre = make_effect(settings.preprocess, settings.preprocess_strength, width, height)
        sr = make_effect(settings.quality, settings.strength, dst_width, dst_height)

        def apply(rgb_bytes: bytes) -> bytes:
            # Copy bytes into writable ndarray; never retain source frames.
            image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
            gpu = torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32)
            gpu = gpu.permute(2, 0, 1).contiguous().div_(255.0)
            stream = torch.cuda.current_stream(device)
            ptr = int(stream.cuda_stream)
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
            del cpu, enhanced, gpu
            return result

        yield apply
    finally:
        if sr is not None:
            sr.close()
        if pre is not None:
            pre.close()
