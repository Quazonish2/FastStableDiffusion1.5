from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image


TEST_DIR = Path(__file__).resolve().parent
INPUT_PATH = TEST_DIR / "VAEencoder_test.jpg"
ENGINE_ROOT = PROJECT_DIR / ".sd_trt_cache"
HEIGHT = 512
WIDTH = 512
WARMUP_RUNS = 3
MEASURED_RUNS = 10


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def find_tiny_vae_encoder() -> tuple[Path, Path]:
    candidates: list[tuple[float, Path, Path]] = []
    for model_path in ENGINE_ROOT.glob("*/onnx/vae_encoder/model.onnx"):
        config_path = model_path.parent / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            block_channels = tuple(config.get("encoder_block_out_channels", ()))
            is_tiny_vae = (
                int(config.get("latent_channels", -1)) == 4
                and abs(float(config.get("scaling_factor", 0.0)) - 1.0) < 1e-6
                and len(block_channels) == 4
                and all(int(value) == 64 for value in block_channels)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if is_tiny_vae:
            candidates.append((model_path.stat().st_mtime, model_path, config_path))

    if not candidates:
        raise FileNotFoundError(
            "Tiny VAE TensorRT ONNX encoder was not found in .sd_trt_cache. "
            "Run txt2img_example.py or img2img_example.py once to export it."
        )
    _, model_path, config_path = max(candidates, key=lambda item: item[0])
    return model_path, config_path


def make_provider_options(cache_root: Path) -> tuple[list[str], list[dict[str, object]]]:
    device_id = torch.cuda.current_device()
    engine_cache = cache_root / "trt_engines"
    engine_cache.mkdir(parents=True, exist_ok=True)
    trt_options: dict[str, object] = {
        "device_id": device_id,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(engine_cache),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_root / "timing.cache"),
        "trt_max_workspace_size": 256 * 1024 * 1024,
        "trt_builder_optimization_level": 2,
        "trt_auxiliary_streams": 0,
    }
    cuda_options: dict[str, object] = {
        "device_id": device_id,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return providers, [trt_options, cuda_options, {}]


def load_image_tensor(path: Path, *, dtype: torch.dtype) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    array = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...] / 127.5 - 1.0)
    return torch.from_numpy(array).to(device="cuda", dtype=dtype)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this TensorRT VAE encoder benchmark.")
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input image not found: {INPUT_PATH}")
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "ONNX Runtime has no TensorrtExecutionProvider. "
            f"Available providers: {ort.get_available_providers()}"
        )

    model_path, config_path = find_tiny_vae_encoder()
    cache_root = model_path.parents[2]
    providers, provider_options = make_provider_options(cache_root)
    ort.set_default_logger_severity(2)

    print("=== Benchmark Tiny VAE encoder ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Input: {INPUT_PATH}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Encoder ONNX: {model_path}")
    print(f"Encoder config: {config_path}")
    print(f"Warm-up runs: {WARMUP_RUNS}; measured runs: {MEASURED_RUNS}")
    print("Loaded components: VAE encoder only; UNet/text encoder/decoder are not loaded.")

    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
        provider_options=provider_options,
    )
    load_time = time.perf_counter() - started
    print(f"Session initialization: {load_time:.3f} s")
    print(f"Execution providers: {session.get_providers()}")

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_dtype = torch.float16 if "float16" in input_meta.type else torch.float32
    output_dtype = np.float16 if "float16" in output_meta.type else np.float32
    input_tensor = load_image_tensor(INPUT_PATH, dtype=input_dtype)
    output_tensor = torch.empty((1, 4, HEIGHT // 8, WIDTH // 8), device="cuda", dtype=input_dtype)

    io_binding = session.io_binding()
    io_binding.bind_input(
        name=input_meta.name,
        device_type="cuda",
        device_id=torch.cuda.current_device(),
        element_type=np.float16 if input_dtype == torch.float16 else np.float32,
        shape=tuple(input_tensor.shape),
        buffer_ptr=input_tensor.data_ptr(),
    )
    io_binding.bind_output(
        name=output_meta.name,
        device_type="cuda",
        device_id=torch.cuda.current_device(),
        element_type=output_dtype,
        shape=tuple(output_tensor.shape),
        buffer_ptr=output_tensor.data_ptr(),
    )

    def run_encoder() -> None:
        session.run_with_iobinding(io_binding)

    for index in range(WARMUP_RUNS):
        sync_cuda()
        run_encoder()
        sync_cuda()
        print(f"Warm-up {index + 1}/{WARMUP_RUNS} complete")

    timings: list[float] = []
    for index in range(MEASURED_RUNS):
        sync_cuda()
        started = time.perf_counter()
        run_encoder()
        sync_cuda()
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        print(f"run={index + 1:02d}: {elapsed * 1000:.3f} ms")

    average = statistics.mean(timings)
    print("\n=== Result ===")
    print(f"Average encoder time: {average * 1000:.3f} ms")
    print(f"Minimum / maximum:    {min(timings) * 1000:.3f} / {max(timings) * 1000:.3f} ms")
    print(f"Throughput:           {1.0 / average:.2f} encodes/s")
    print(f"Output latent shape:  {tuple(output_tensor.shape)}")


if __name__ == "__main__":
    main()
