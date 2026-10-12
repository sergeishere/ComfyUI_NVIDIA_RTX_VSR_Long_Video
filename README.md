# ComfyUI NVIDIA RTX VSR Long Video

NVIDIA RTX Video Super Resolution for ComfyUI, optimized for **long videos with low RAM usage**.

Uses NVIDIA's `nvidia-vfx` SDK to upscale, denoise, and deblur videos without loading the entire video into memory.

<img width="1278" height="1068" alt="image" src="https://github.com/user-attachments/assets/baec5de4-0523-44e0-b4e4-ac3fa4c11d0b" />

## Features

- **NVIDIA RTX Video Super Resolution** — AI upscaling up to 4×
- **Deblur & Denoise** — optional preprocessing or standalone enhancement
- **Low RAM usage** — frame-by-frame processing with temporary disk segments
- **Native VIDEO input/output** — compatible with ComfyUI's video nodes
- **Built-in video preview** — playback directly inside the node
- **H.264 / HEVC NVENC** — GPU-accelerated video encoding
- **Audio preservation** — retains the original audio
- **Automatic cleanup** — removes temporary files after processing and abandoned files on the next run

## Installation

Clone this repository into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/tritant/ComfyUI_NVIDIA_RTX_VSR_Long_Video.git
```

Install the dependencies using ComfyUI's Python environment:

```bash
python -m pip install -r requirements.txt
```

**Requirements:** NVIDIA RTX GPU, compatible CUDA/PyTorch environment, `nvidia-vfx >= 0.2`, and FFmpeg.

For NVENC encoding, FFmpeg must support `h264_nvenc` or `hevc_nvenc`.

## Usage

Connect a native **Load Video** node or any compatible `VIDEO` output to **RTX VSR Long Video**.

```text
Load Video
    ↓
RTX VSR Long Video
    ↓
VIDEO output → Other video nodes
```

Select your upscale factor, NVIDIA quality mode, strength, and optional Deblur/Denoise preprocessing.

The finished MP4 is automatically saved to:

```text
ComfyUI/output/NVIDIA_RTX_VSR_Long_Video/
```

The node also provides an integrated preview, a file path output, and a native `VIDEO` output for further processing.

## How It Works

Instead of keeping thousands of uncompressed frames in RAM, the node:

1. Decodes and processes video frames progressively.
2. Applies NVIDIA RTX VSR using the GPU.
3. Encodes temporary video segments to disk.
4. Assembles the final MP4 without a second video encoding pass.
5. Preserves audio and automatically cleans up temporary files.

**Memory usage stays manageable even with long, high-resolution videos.**

## Notes

- H.264 NVENC supports output dimensions up to 4096 pixels per axis. Use HEVC NVENC for larger resolutions.
- Currently supports SDR 8-bit video processing.
- HEVC playback in the integrated preview depends on browser support.

## Credits

Powered by [NVIDIA Video Effects SDK](https://docs.nvidia.com/maxine/vfx-python/latest/api.html).

NVIDIA RTX VSR technology and models belong to NVIDIA. This project is an independent ComfyUI integration.
