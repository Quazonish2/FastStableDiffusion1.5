from __future__ import annotations

import gc
import logging
import statistics
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

PROCESS_STARTED = time.perf_counter()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


class _IgnoreKnownWarning(logging.Filter):
    _fragments = (
        "Multiple distributions found for package optimum",
        "No LoRA keys associated to CLIPTextModel",
        "You have disabled the safety checker",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in self._fragments)


_known_warning_filter = _IgnoreKnownWarning()
for _logger_name in (
    "diffusers.utils.import_utils",
    "diffusers.loaders.lora_base",
    "diffusers.loaders.peft",
    "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
):
    logging.getLogger(_logger_name).addFilter(_known_warning_filter)

warnings.filterwarnings(
    "ignore",
    message=r"Accessing config attribute `num_hidden_layers` directly.*",
    category=FutureWarning,
    module=r"diffusers\.configuration_utils",
)

import torch
from fast_stable_diffusion import FastStableDiffusion, LoRA

IMPORTS_FINISHED = time.perf_counter()
BASE_DIR = PROJECT_DIR
MODEL_PATH = BASE_DIR / "models" / "sd" / "v1-5-pruned-emaonly.safetensors"
LORA_PATH = BASE_DIR / "models" / "lora" / "Hyper-SD15-2steps-lora.safetensors"
TINY_VAE_PATH = BASE_DIR / "models" / "vae"
ENGINE_DIR = BASE_DIR / ".sd_trt_cache"
PROMPT = "a photo of an astronaut riding a horse on mars"
HEIGHT = 512
WIDTH = 512
STEPS = 1
GUIDANCE_SCALE = 1.0
SEEDS = tuple(range(10))


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"


def format_mb(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 * 1024):.0f} MiB"


def memory_snapshot() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }


@dataclass
class StageTimer:
    records: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def patch_method(self, owner: type[Any], method_name: str, label: str) -> None:
        original = getattr(owner, method_name)

        @wraps(original)
        def timed(instance: Any, *args: Any, **kwargs: Any) -> Any:
            print(f"[stage] {label} ...", flush=True)
            sync_cuda()
            started = time.perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                sync_cuda()
                elapsed = time.perf_counter() - started
                self.records[label].append(elapsed)
                print(f"[stage] {label}: {format_seconds(elapsed)}", flush=True)

        setattr(owner, method_name, timed)


@dataclass
class SampleTiming:
    values: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    wall_time: float = 0.0
    peak_allocated: int | None = None
    peak_reserved: int | None = None


class ComponentTimer:
    def __init__(self) -> None:
        self.current_values: dict[str, float] | None = None
        self.current_calls: dict[str, int] | None = None
        self.samples: list[SampleTiming] = []

    def begin(self) -> None:
        self.current_values = defaultdict(float)
        self.current_calls = defaultdict(int)

    def add(self, label: str, elapsed: float) -> None:
        if self.current_values is None or self.current_calls is None:
            return
        self.current_values[label] += elapsed
        self.current_calls[label] += 1

    def finish(
        self,
        wall_time: float,
        snapshot: dict[str, int] | None,
    ) -> SampleTiming:
        sample = SampleTiming(
            values=dict(self.current_values or {}),
            calls=dict(self.current_calls or {}),
            wall_time=wall_time,
            peak_allocated=snapshot["peak_allocated"] if snapshot else None,
            peak_reserved=snapshot["peak_reserved"] if snapshot else None,
        )
        self.samples.append(sample)
        self.current_values = None
        self.current_calls = None
        return sample

    def clear(self) -> None:
        self.samples.clear()


def wrap_instance_method(
    instance: Any,
    method_name: str,
    label: str,
    timer: ComponentTimer,
) -> bool:
    original = getattr(instance, method_name, None)
    if original is None or not callable(original):
        return False

    @wraps(original)
    def timed(*args: Any, **kwargs: Any) -> Any:
        sync_cuda()
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            sync_cuda()
            timer.add(label, time.perf_counter() - started)

    try:
        setattr(instance, method_name, timed)
    except (AttributeError, TypeError):
        return False
    return True


def install_component_timers(
    pipeline: Any,
    timer: ComponentTimer,
    already_installed: set[int],
) -> list[str]:
    pipeline_id = id(pipeline)
    if pipeline_id in already_installed:
        return []
    already_installed.add(pipeline_id)

    installed: list[str] = []

    targets: list[tuple[Any, str, str]] = [
        (pipeline, "encode_prompt", "prompt encoding"),
        (pipeline, "prepare_latents", "latent preparation"),
        (pipeline.scheduler, "set_timesteps", "scheduler setup"),
        (pipeline.scheduler, "scale_model_input", "scheduler scale input"),
        (pipeline.scheduler, "step", "scheduler step"),
        (pipeline.unet, "forward", "UNet"),
        (pipeline, "run_safety_checker", "safety checker"),
    ]

    text_encoder = getattr(pipeline, "text_encoder", None)
    if text_encoder is not None:
        targets.append((text_encoder, "forward", "text encoder"))

    vae_decoder = getattr(pipeline, "vae_decoder", None)
    if vae_decoder is not None and hasattr(vae_decoder, "forward"):
        targets.append((vae_decoder, "forward", "VAE decoder"))
    else:
        targets.append((pipeline.vae, "decode", "VAE decode"))

    image_processor = getattr(pipeline, "image_processor", None)
    if image_processor is not None:
        targets.append((image_processor, "postprocess", "image postprocess"))

    for instance, method_name, label in targets:
        if instance is not None and wrap_instance_method(
            instance,
            method_name,
            label,
            timer,
        ):
            installed.append(label)

    return installed


def average(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def print_stage_summary(stage_timer: StageTimer, construction_time: float) -> None:
    print("\n=== Loading and initialization ===")
    print(
        f"Library imports:               "
        f"{format_seconds(IMPORTS_FINISHED - PROCESS_STARTED)}"
    )
    print(
        f"FastStableDiffusion.__init__: "
        f"{format_seconds(construction_time)}"
    )

    for label, values in stage_timer.records.items():
        if len(values) == 1:
            print(f"{label + ':':32} {format_seconds(values[0])}")
        else:
            joined = ", ".join(format_seconds(value) for value in values)
            print(f"{label + ':':32} {joined}")


def print_memory(label: str, snapshot: dict[str, int] | None) -> None:
    if snapshot is None:
        print(f"CUDA memory ({label}):          unavailable")
        return

    print(
        f"CUDA memory ({label}):          "
        f"allocated={format_mb(snapshot['allocated'])}, "
        f"reserved={format_mb(snapshot['reserved'])}"
    )


def print_sample_table(samples: list[SampleTiming]) -> None:
    print("\n=== Individual generations ===")
    print("seed | total       | prompt      | UNet        | scheduler   | VAE")
    print("-----+-------------+-------------+-------------+-------------+-------------")

    for seed, sample in zip(SEEDS, samples, strict=True):
        print(
            f"{seed:4d} | {format_seconds(sample.wall_time):11} | "
            f"{format_seconds(sample.values.get('prompt encoding')):11} | "
            f"{format_seconds(sample.values.get('UNet')):11} | "
            f"{format_seconds(sample.values.get('scheduler step')):11} | "
            f"{format_seconds(sample.values.get('VAE decoder') or sample.values.get('VAE decode')):11}"
        )


def print_component_summary(samples: list[SampleTiming]) -> None:
    print("\n=== Average timings for 10 images ===")
    print(f"Total images:                  {len(samples)}")
    print(
        f"Average total generation:      "
        f"{format_seconds(average([x.wall_time for x in samples]))}"
    )
    print(
        f"Minimum / maximum:             "
        f"{format_seconds(min(x.wall_time for x in samples))} / "
        f"{format_seconds(max(x.wall_time for x in samples))}"
    )

    labels = (
        "prompt encoding",
        "text encoder",
        "latent preparation",
        "scheduler setup",
        "scheduler scale input",
        "UNet",
        "scheduler step",
        "VAE decoder",
        "VAE decode",
        "safety checker",
        "image postprocess",
    )

    for label in labels:
        values = [
            sample.values[label]
            for sample in samples
            if label in sample.values
        ]
        if not values:
            continue

        calls = sum(
            sample.calls.get(label, 0)
            for sample in samples
        )

        print(
            f"{label + ':':32} "
            f"avg={format_seconds(average(values))}, "
            f"calls/image={calls / len(samples):.2f}"
        )

    phase_labels = (
        "prompt encoding",
        "latent preparation",
        "scheduler setup",
        "scheduler scale input",
        "UNet",
        "scheduler step",
        "VAE decoder",
        "VAE decode",
        "safety checker",
        "image postprocess",
    )

    residuals: list[float] = []

    for sample in samples:
        accounted = sum(
            sample.values.get(label, 0.0)
            for label in phase_labels
        )
        residuals.append(sample.wall_time - accounted)

    print(
        f"Pipeline/other residual:      "
        f"{format_seconds(average(residuals))}"
    )

    peak_allocated = [
        x.peak_allocated
        for x in samples
        if x.peak_allocated is not None
    ]
    peak_reserved = [
        x.peak_reserved
        for x in samples
        if x.peak_reserved is not None
    ]

    if peak_allocated:
        print(
            f"Peak allocated per run:       "
            f"{format_mb(max(peak_allocated))}"
        )
        print(
            f"Peak reserved per run:        "
            f"{format_mb(max(peak_reserved))}"
        )


def main() -> None:
    print("=== Stable Diffusion Diagnostics ===")
    print(
        f"Resolution: {WIDTH}x{HEIGHT}, "
        f"steps: {STEPS}, "
        f"guidance scale: {GUIDANCE_SCALE}"
    )
    print(
        f"Seeds: {SEEDS[0]}..{SEEDS[-1]}; "
        f"images are not saved"
    )
    print(
        "Safety checker: intentionally disabled "
        "for local diagnostics"
    )
    print(f"CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    print(f"Cache: {ENGINE_DIR}")

    stage_timer = StageTimer()

    stage_timer.patch_method(
        FastStableDiffusion,
        "_load_native_pipeline",
        "Native model pipeline load",
    )
    stage_timer.patch_method(
        FastStableDiffusion,
        "_load_and_apply_loras",
        "LoRA load and fuse",
    )
    stage_timer.patch_method(
        FastStableDiffusion,
        "_load_tensorrt_pipeline",
        "TensorRT/ONNX pipeline init",
    )

    print(
        "\n[stage] FastStableDiffusion initialization ...",
        flush=True,
    )

    sync_cuda()
    construction_started = time.perf_counter()

    generator = FastStableDiffusion(
        model_id=MODEL_PATH,
        backend="auto",
        loras=[LoRA(LORA_PATH)],
        hyper_sd=True,
        vae_model_id=TINY_VAE_PATH,
        engine_dir=ENGINE_DIR,
        disable_safety_checker=True,
        show_progress=False,
    )

    sync_cuda()
    construction_time = time.perf_counter() - construction_started

    print(
        "[stage] FastStableDiffusion initialization: "
        f"{format_seconds(construction_time)}",
        flush=True,
    )
    print(f"Backend selected: {generator.backend}")
    print_memory("after loading", memory_snapshot())

    component_timer = ComponentTimer()
    installed_pipeline_ids: set[int] = set()

    installed = install_component_timers(
        generator.pipeline,
        component_timer,
        installed_pipeline_ids,
    )

    print(
        f"Components being monitored: "
        f"{', '.join(installed) or 'no available hooks'}"
    )

    print("\n[stage] Warm-up ...", flush=True)

    component_timer.begin()
    sync_cuda()
    warmup_started = time.perf_counter()

    generator.warmup(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        use_tiny_vae=True,
    )

    sync_cuda()
    warmup_time = time.perf_counter() - warmup_started

    warmup_sample = component_timer.finish(
        warmup_time,
        memory_snapshot(),
    )

    print(
        f"[stage] Warm-up: {format_seconds(warmup_time)}",
        flush=True,
    )
    print_memory("after warm-up", memory_snapshot())

    installed_after_warmup = install_component_timers(
        generator.pipeline,
        component_timer,
        installed_pipeline_ids,
    )

    if installed_after_warmup:
        print(
            "Hooks after backend switch: "
            f"{', '.join(installed_after_warmup)}"
        )

    component_timer.clear()

    print(
        "\n[stage] 10 measured generations ...",
        flush=True,
    )

    for seed in SEEDS:
        component_timer.begin()
        sync_cuda()
        started = time.perf_counter()

        image = generator.generate(
            PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=seed,
            use_tiny_vae=True,
        )

        sync_cuda()
        elapsed = time.perf_counter() - started

        sample = component_timer.finish(
            elapsed,
            memory_snapshot(),
        )

        del image

        print(
            f"seed={seed}: {format_seconds(sample.wall_time)}",
            flush=True,
        )

    measured_samples = list(component_timer.samples)

    print_sample_table(measured_samples)
    print_component_summary(measured_samples)

    print("\n=== Warm-up and memory ===")
    print(
        f"Warm-up total:                 "
        f"{format_seconds(warmup_sample.wall_time)}"
    )

    for label, elapsed in warmup_sample.values.items():
        print(
            f"Warm-up {label + ':':24} "
            f"{format_seconds(elapsed)}"
        )

    print_memory("after 10 generations", memory_snapshot())

    print(
        f"Total benchmark_everything.py time: "
        f"{format_seconds(time.perf_counter() - PROCESS_STARTED)}"
    )

    gc.collect()


if __name__ == "__main__":
    main()
