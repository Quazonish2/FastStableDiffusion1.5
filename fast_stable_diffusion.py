from __future__ import annotations
import gc
import hashlib
import json
import logging
import math
import os
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from onnxruntime import set_default_logger_severity
set_default_logger_severity(3)

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

LOGGER = logging.getLogger(__name__)

for _logger_name in (
    "diffusers",
    "transformers",
    "optimum",
    "huggingface_hub",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

for _warning_category in (DeprecationWarning, FutureWarning, UserWarning):
    warnings.filterwarnings(
        "ignore",
        category=_warning_category,
        module=r"(diffusers|transformers|optimum|huggingface_hub)(\\..*)?",
    )

_PROJECT_DIR = Path(__file__).resolve().parent
_LOCAL_MODEL_PATH = _PROJECT_DIR / "models" / "sd" / "v1-5-pruned-emaonly.safetensors"
_LOCAL_TINY_VAE_PATH = _PROJECT_DIR / "models" / "vae"
DEFAULT_MODEL_ID = str(_LOCAL_MODEL_PATH) if _LOCAL_MODEL_PATH.is_file() else "sd-legacy/stable-diffusion-v1-5"
DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"
DEFAULT_TINY_VAE_ID = (
    str(_LOCAL_TINY_VAE_PATH)
    if (_LOCAL_TINY_VAE_PATH / "config.json").is_file()
    else "madebyollin/taesd"
)
_CACHE_FORMAT_VERSION = "sd15-trt-v3"
_MIN_TRT_WORKSPACE_SIZE = 256 * 1024 * 1024
_MAX_TRT_WORKSPACE_SIZE = 2 * 1024 * 1024 * 1024
_TRT_WORKSPACE_FRACTION = 0.125
_CUDA_GRAPH_TRT_WORKSPACE_SIZE = 768 * 1024 * 1024
_VRAM_GB = 1024 ** 3

@dataclass(frozen=True)
class LoRA:
    path: str | os.PathLike[str]
    scale: float = 1.0
    adapter_name: str | None = None
    weight_name: str | None = None

@dataclass(frozen=True)
class _NormalizedLoRA:
    path: str
    scale: float
    adapter_name: str
    weight_name: str | None

def _normalize_lora(value: LoRA | Mapping[str, Any] | str | os.PathLike[str]) -> LoRA:
    if isinstance(value, LoRA):
        return value
    if isinstance(value, Mapping):
        if "path" not in value:
            raise ValueError("Each LoRA mapping must contain a 'path' key.")
        return LoRA(
            path=value["path"],
            scale=float(value.get("scale", 1.0)),
            adapter_name=value.get("adapter_name"),
            weight_name=value.get("weight_name"),
        )
    return LoRA(path=value)

def _normalize_loras(
    loras: Sequence[LoRA | Mapping[str, Any] | str | os.PathLike[str]] | None,
) -> tuple[_NormalizedLoRA, ...]:
    if not loras:
        return ()

    normalized: list[_NormalizedLoRA] = []
    names: set[str] = set()
    for index, raw_lora in enumerate(loras):
        lora = _normalize_lora(raw_lora)
        scale = float(lora.scale)
        if not scale == scale or scale in (float("inf"), float("-inf")):
            raise ValueError(f"Invalid LoRA scale: {scale!r}")

        name = lora.adapter_name or f"lora_{index}"
        if name in names:
            raise ValueError(f"Duplicate LoRA adapter_name: {name!r}")
        names.add(name)
        normalized.append(
            _NormalizedLoRA(
                path=os.fspath(lora.path),
                scale=scale,
                adapter_name=name,
                weight_name=lora.weight_name,
            )
        )
    return tuple(normalized)

def _is_single_file_source(model_id: str | os.PathLike[str]) -> bool:
    path = Path(model_id).expanduser()
    return path.is_file() or path.suffix.lower() in {".safetensors", ".ckpt", ".pt"}

def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    current = torch.cuda.current_device()
    return int(current)

def _safe_file_fingerprint(path: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return f"remote:{path}"
    try:
        stat = candidate.stat()
        return f"file:{candidate.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return f"file:{path}"

@dataclass
class _TinyLatentDistribution:
    parameters: torch.Tensor

class _TinyEncodeOutput:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latent_dist = _TinyLatentDistribution(parameters=latents)

    def __getitem__(self, key: str) -> _TinyLatentDistribution:
        if key != "latent_dist":
            raise KeyError(key)
        return self.latent_dist

class _TinyVaeExportConfig(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self)

    def save_pretrained(self, save_directory: str | os.PathLike[str]) -> None:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(dict(self), handle, ensure_ascii=False, indent=2)

class _TinyVaeExportAdapter(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.inner = vae

        config_values = dict(vae.config)
        block_count = len(config_values.get("encoder_block_out_channels", (64, 64, 64, 64)))
        config_values.setdefault("sample_size", 32)
        config_values["down_block_types"] = tuple("DownEncoderBlock2D" for _ in range(block_count))
        config_values["up_block_types"] = tuple("UpDecoderBlock2D" for _ in range(block_count))
        self.config = _TinyVaeExportConfig(config_values)

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> _TinyEncodeOutput:
        output = self.inner.encode(x, return_dict=True)
        adapted = _TinyEncodeOutput(output.latents)
        return adapted if return_dict else adapted.latent_dist.parameters

    def decode(
        self,
        z: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        latent_sample = z if z is not None else x
        if latent_sample is None:
            raise ValueError("TAESD export decode requires a latent tensor.")
        return self.inner.decode(latent_sample, return_dict=return_dict)

class _SuppressExportLogMessages(logging.Filter):
    _fragments = (
        "Expected types for vae:",
        "Keyword arguments {'subfolder': '', 'trust_remote_code': False}",
        "were passed to AutoencoderTiny, but are not expected",
        "You have disabled the safety checker for",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in self._fragments)

@contextmanager
def _quiet_onnx_export_warnings() -> Any:
    logger_names = (
        "diffusers.pipelines.pipeline_utils",
        "diffusers.configuration_utils",
    )
    log_filter = _SuppressExportLogMessages()
    loggers = [logging.getLogger(name) for name in logger_names]
    for logger in loggers:
        logger.addFilter(log_filter)

    with warnings.catch_warnings():
        tracer_warning = getattr(torch.jit, "TracerWarning", UserWarning)
        warnings.filterwarnings("ignore", category=tracer_warning)
        warnings.filterwarnings(
            "ignore",
            message=r"Exporting aten::index operator of advanced indexing.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Constant folding - Only steps=1 can be constant folded.*",
            category=UserWarning,
        )
        try:
            yield
        finally:
            for logger in loggers:
                logger.removeFilter(log_filter)

@contextmanager
def _tiny_vae_export_compatibility(enabled: bool) -> Any:
    if not enabled:
        yield
        return

    try:
        from diffusers import AutoencoderTiny
        import optimum.exporters.utils as optimum_export_utils
    except ImportError:
        yield
        return

    original = optimum_export_utils.get_diffusion_models_for_export

    def patched_get_diffusion_models_for_export(pipeline: Any, *args: Any, **kwargs: Any) -> Any:
        vae = getattr(pipeline, "vae", None)
        if not isinstance(vae, AutoencoderTiny):
            return original(pipeline, *args, **kwargs)

        pipeline.vae = _TinyVaeExportAdapter(vae)
        try:
            return original(pipeline, *args, **kwargs)
        finally:
            pipeline.vae = vae

    optimum_export_utils.get_diffusion_models_for_export = patched_get_diffusion_models_for_export
    try:
        yield
    finally:
        optimum_export_utils.get_diffusion_models_for_export = original

class FastStableDiffusion:
    def __init__(
        self,
        model_id: str | os.PathLike[str] = DEFAULT_MODEL_ID,
        *,
        backend: str = "auto",
        use_tensorrt: bool = True,
        engine_dir: str | os.PathLike[str] = ".sd_trt_cache",
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        loras: Sequence[LoRA | Mapping[str, Any] | str | os.PathLike[str]] | None = None,
        fuse_loras: bool = True,
        use_xformers: bool = True,
        channels_last: bool | None = None,
        compile_unet: bool = False,
        compile_vae: bool = False,
        compile_mode: str = "reduce-overhead",
        disable_safety_checker: bool = False,
        vae_slicing: bool | None = None,
        vae_tiling: bool | None = None,
        memory_mode: str = "auto",
        hyper_sd: bool = False,
        vae_model_id: str | os.PathLike[str] | None = None,
        tiny_vae_model_id: str | os.PathLike[str] | None = None,
        warmup_on_init: bool = False,
        warmup_height: int = 512,
        warmup_width: int = 512,
        warmup_steps: int = 1,
        warmup_guidance_scale: float | None = None,
        show_progress: bool = False,
        model_config: str | os.PathLike[str] | None = None,
        original_config_file: str | os.PathLike[str] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        trt_provider_options: Mapping[str, Any] | None = None,
        cache_tag: str | None = None,
    ) -> None:
        self.model_id = os.fspath(model_id)
        self.requested_backend = backend.lower().replace("-", "_")
        if self.requested_backend == "trt":
            self.requested_backend = "tensorrt"
        if self.requested_backend not in {"auto", "torch", "diffusers", "tensorrt", "onnxruntime"}:
            raise ValueError(
                "backend must be one of 'auto', 'torch', 'diffusers', 'tensorrt', or 'onnxruntime'."
            )
        if self.requested_backend == "onnxruntime":
            self.requested_backend = "tensorrt"

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            if self.requested_backend == "tensorrt":
                raise RuntimeError("TensorRT backend requires a CUDA device, but torch.cuda is unavailable.")
            warnings.warn("CUDA is unavailable; using CPU Diffusers inference.", RuntimeWarning, stacklevel=2)
            requested_device = torch.device("cpu")
        self.device = requested_device

        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.device.type == "cpu" and dtype != torch.float32:
            warnings.warn("CPU inference uses float32 for compatibility.", RuntimeWarning, stacklevel=2)
            dtype = torch.float32
        if dtype not in {torch.float16, torch.float32, torch.bfloat16}:
            raise ValueError("dtype must be torch.float16, torch.float32, or torch.bfloat16.")
        self.dtype = dtype

        self.engine_dir = Path(engine_dir).expanduser()
        self.loras = _normalize_loras(loras)
        self.fuse_loras = bool(fuse_loras)
        self.use_tensorrt = bool(use_tensorrt)
        self.use_xformers = bool(use_xformers)
        self.channels_last = channels_last
        self.compile_unet = bool(compile_unet)
        self.compile_vae = bool(compile_vae)
        self.compile_mode = compile_mode
        self.disable_safety_checker = bool(disable_safety_checker)
        self.vae_slicing = vae_slicing
        self.vae_tiling = vae_tiling
        self.memory_mode = memory_mode.lower().replace("-", "_")
        if self.memory_mode not in {"auto", "low", "balanced", "high", "performance"}:
            raise ValueError(
                "memory_mode must be one of 'auto', 'low', 'balanced', 'high', or 'performance'."
            )
        self.hyper_sd = bool(hyper_sd)
        self.tiny_vae_model_id = os.fspath(
            tiny_vae_model_id if tiny_vae_model_id is not None else (vae_model_id or DEFAULT_TINY_VAE_ID)
        )
        self.vae_model_id = os.fspath(vae_model_id) if vae_model_id is not None else self.tiny_vae_model_id
        self.show_progress = bool(show_progress)
        self.model_config = os.fspath(model_config) if model_config is not None else None
        self.original_config_file = (
            os.fspath(original_config_file) if original_config_file is not None else None
        )
        self.model_kwargs = dict(model_kwargs or {})
        self.trt_provider_options = dict(trt_provider_options or {})
        self.cache_tag = cache_tag
        self._pipeline: Any = None
        self._backend = ""
        self._lora_fused = False
        self._active_lora_names: tuple[str, ...] = ()
        self._lock = threading.RLock()

        self._configure_cuda()
        self._load()

        if warmup_on_init:
            self.warmup(
                height=warmup_height,
                width=warmup_width,
                num_inference_steps=warmup_steps,
                guidance_scale=warmup_guidance_scale,
            )

    @property
    def pipeline(self) -> Any:
            return self._pipeline

    @property
    def backend(self) -> str:
            return self._backend

    @property
    def torch_device(self) -> torch.device:
        return self.device

    def _configure_cuda(self) -> None:
        if self.device.type != "cuda":
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass

    def _vram_info(self) -> tuple[int, int]:
        if self.device.type != "cuda":
            return 0, 0
        device_index = _device_index(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        return int(free_bytes), int(total_bytes)

    def _memory_tier(self) -> str:
        if self.device.type != "cuda":
            return "cpu"
        _, total_bytes = self._vram_info()
        total_gb = total_bytes / _VRAM_GB
        if self.memory_mode != "auto":
            return self.memory_mode
        if total_gb < 4.5:
            return "low"
        if total_gb < 8.0:
            return "balanced"
        if total_gb < 12.0:
            return "high"
        return "performance"

    def _auto_channels_last(self) -> bool:
        if self.channels_last is not None:
            return bool(self.channels_last)
        if self.device.type != "cuda":
            return False
        _, total_bytes = self._vram_info()
        return total_bytes >= int(4.5 * _VRAM_GB)

    def _trt_workspace_size(self) -> int:
        if self.device.type != "cuda":
            return _MIN_TRT_WORKSPACE_SIZE

        free_bytes, total_bytes = self._vram_info()
        target = int(total_bytes * _TRT_WORKSPACE_FRACTION)
        target = max(_MIN_TRT_WORKSPACE_SIZE, min(target, _MAX_TRT_WORKSPACE_SIZE))

        reserve = max(512 * 1024 * 1024, min(int(total_bytes * 0.10), 1536 * 1024 * 1024))
        available_for_workspace = max(0, free_bytes - reserve)
        if available_for_workspace >= _MIN_TRT_WORKSPACE_SIZE:
            target = min(target, available_for_workspace)

        return max(_MIN_TRT_WORKSPACE_SIZE, target)

    def _auto_use_tiny_vae(self) -> bool:
        return True

    def _configure_native_request_memory(
        self,
        *,
        height: int | None,
        width: int | None,
        num_images_per_prompt: int,
    ) -> None:
        if self._backend != "torch":
            return

        pipe = self.pipeline
        if pipe is None or height is None or width is None:
            return

        pixels = int(height) * int(width)
        tier = self._memory_tier()
        total_gb = 0.0
        if self.device.type == "cuda":
            _, total_bytes = self._vram_info()
            total_gb = total_bytes / _VRAM_GB

        if self.vae_slicing is None:
            if tier == "low":
                use_slicing = num_images_per_prompt > 1
            elif tier == "balanced":
                use_slicing = num_images_per_prompt > 1
            elif tier == "high":
                use_slicing = num_images_per_prompt > 2
            else:
                use_slicing = num_images_per_prompt > 4
        else:
            use_slicing = bool(self.vae_slicing)

        if self.vae_tiling is None:
            if tier == "low" or total_gb < 6.0:
                use_tiling = pixels > 512 * 512
            elif tier == "balanced":
                use_tiling = pixels > 768 * 768
            elif tier == "high":
                use_tiling = pixels > 1024 * 1024
            else:
                use_tiling = pixels > 1536 * 1536
        else:
            use_tiling = bool(self.vae_tiling)

        if hasattr(pipe, "enable_vae_slicing") and hasattr(pipe, "disable_vae_slicing"):
            (pipe.enable_vae_slicing if use_slicing else pipe.disable_vae_slicing)()
        if hasattr(pipe, "enable_vae_tiling") and hasattr(pipe, "disable_vae_tiling"):
            (pipe.enable_vae_tiling if use_tiling else pipe.disable_vae_tiling)()

    def _load(self) -> None:
        wants_tensorrt = self.requested_backend == "tensorrt" or (
            self.requested_backend == "auto" and self.use_tensorrt
        )
        if wants_tensorrt and self.device.type == "cuda":
            if self._tensorrt_available():
                try:
                    self._pipeline = self._load_tensorrt_pipeline()
                    self._backend = "tensorrt"
                    LOGGER.info("Using ONNX Runtime TensorRT backend.")
                    return
                except Exception as exc:
                    if self.requested_backend == "tensorrt":
                        raise RuntimeError(
                            "TensorRT pipeline initialization failed. "
                            f"Use backend='auto' for a Diffusers fallback. Original error: {exc}"
                        ) from exc
                    warnings.warn(
                        f"TensorRT initialization failed ({exc!r}); falling back to Diffusers CUDA.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._release_pipeline()
            elif self.requested_backend == "tensorrt":
                raise RuntimeError(
                    "ONNX Runtime has no TensorrtExecutionProvider. "
                    "Install a compatible onnxruntime-gpu/TensorRT stack."
                )

        self._pipeline = self._load_native_pipeline(with_loras=True, for_export=False)
        self._backend = "torch"

    @staticmethod
    def _tensorrt_available() -> bool:
        try:
            import onnxruntime as ort

            return "TensorrtExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def _load_native_pipeline(self, *, with_loras: bool, for_export: bool) -> Any:
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:              raise RuntimeError("Install diffusers and its dependencies before using this module.") from exc

        kwargs = dict(self.model_kwargs)
        kwargs.setdefault("torch_dtype", self.dtype)
        kwargs.setdefault("use_safetensors", True)
        if self.vae_model_id is not None:
            if "vae" in kwargs:
                raise ValueError("Pass either vae_model_id or model_kwargs['vae'], not both.")
            kwargs["vae"] = self._load_custom_vae()

        if _is_single_file_source(self.model_id):
            if self.original_config_file is not None:
                kwargs.setdefault("original_config_file", self.original_config_file)
            elif self.model_config is not None:
                kwargs.setdefault("config", self.model_config)
            if self.disable_safety_checker:
                                                                                kwargs.setdefault("requires_safety_checker", False)
            pipe = StableDiffusionPipeline.from_single_file(self.model_id, **kwargs)
        else:
            if self.disable_safety_checker:
                kwargs.setdefault("requires_safety_checker", False)
            pipe = StableDiffusionPipeline.from_pretrained(self.model_id, **kwargs)

        if with_loras and self.loras:
            self._load_and_apply_loras(pipe)
        if self.hyper_sd:
            self._configure_hyper_sd_scheduler(pipe)

        if for_export:
            return pipe

        pipe = pipe.to(self.device)
        self._optimize_native_pipeline(pipe)
        return pipe

    def _load_custom_vae(self) -> Any:
        try:
            from diffusers import AutoencoderTiny
        except ImportError as exc:              raise RuntimeError("Install diffusers with AutoencoderTiny support to use vae_model_id.") from exc

        vae = AutoencoderTiny.from_pretrained(
            self.vae_model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        config = vae.config
        latent_channels = int(getattr(config, "latent_channels", -1))
        block_out_channels = tuple(getattr(config, "encoder_block_out_channels", ()))
        upsampling_scaling_factor = int(getattr(config, "upsampling_scaling_factor", 2))
        spatial_scale = upsampling_scaling_factor ** max(len(block_out_channels) - 1, 0)
        scaling_factor = float(getattr(config, "scaling_factor", 0.0))

        if latent_channels != 4 or spatial_scale != 8 or abs(scaling_factor - 1.0) > 1e-6:
            raise ValueError(
                "The custom VAE is not compatible with SD1.5: expected "
                "latent_channels=4, spatial_scale=8 and scaling_factor=1.0, "
                f"got latent_channels={latent_channels}, spatial_scale={spatial_scale}, "
                f"scaling_factor={scaling_factor}. Use 'madebyollin/taesd'."
            )
        return vae

    @staticmethod
    def _configure_hyper_sd_scheduler(pipe: Any) -> None:
        try:
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_order=2,
                solver_type="midpoint",
                lower_order_final=True,
                timestep_spacing="trailing",
            )
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("Could not configure the DPM++ 2M scheduler.") from exc

    def _load_and_apply_loras(self, pipe: Any) -> None:
        names: list[str] = []
        scales: list[float] = []
        for lora in self.loras:
            kwargs: dict[str, Any] = {"adapter_name": lora.adapter_name}
            if lora.weight_name is not None:
                kwargs["weight_name"] = lora.weight_name
            pipe.load_lora_weights(lora.path, **kwargs)
            names.append(lora.adapter_name)
            scales.append(lora.scale)

        self._active_lora_names = tuple(names)
        if not self.fuse_loras:
            if len(names) == 1:
                pipe.set_adapters(names[0], adapter_weights=scales[0])
            else:
                pipe.set_adapters(names, adapter_weights=scales)
            return

        if len(names) == 1:
            pipe.set_adapters(names[0], adapter_weights=scales[0])
        else:
            pipe.set_adapters(names, adapter_weights=scales)

        try:
            pipe.fuse_lora(adapter_names=names, lora_scale=1.0, safe_fusing=True)
        except TypeError:              pipe.fuse_lora(adapter_names=names, lora_scale=1.0)
        pipe.unload_lora_weights()
        self._lora_fused = True

    def _optimize_native_pipeline(self, pipe: Any) -> None:
        if self.disable_safety_checker and hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
            try:
                pipe.register_to_config(requires_safety_checker=False)
            except (AttributeError, TypeError):
                pass

        if self.vae_slicing is True and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if self.vae_tiling is True and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=not self.show_progress)

        if self.device.type != "cuda":
            return

        if self._auto_channels_last():
            for component_name in ("unet", "vae"):
                component = getattr(pipe, component_name, None)
                if component is None:
                    continue
                try:
                    component.to(memory_format=torch.channels_last)
                except (AttributeError, RuntimeError):
                    LOGGER.debug("Could not enable channels_last for %s.", component_name, exc_info=True)

        if self.use_xformers:
            try:
                import xformers  
                pipe.enable_xformers_memory_efficient_attention()
            except (ImportError, ModuleNotFoundError):
                                LOGGER.debug("xFormers is not installed; using PyTorch SDPA attention.")
            except Exception:
                LOGGER.debug("xFormers attention could not be enabled; using the default processor.", exc_info=True)

        if self.compile_unet:
            try:
                pipe.unet = torch.compile(pipe.unet, mode=self.compile_mode, fullgraph=False)
            except Exception as exc:
                warnings.warn(f"UNet torch.compile failed; continuing without it ({exc!r}).", RuntimeWarning)

        if self.compile_vae:
            try:
                pipe.vae = torch.compile(pipe.vae, mode=self.compile_mode, fullgraph=False)
            except Exception as exc:
                warnings.warn(f"VAE torch.compile failed; continuing without it ({exc!r}).", RuntimeWarning)

    def _cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(_CACHE_FORMAT_VERSION.encode("utf-8"))
        digest.update(self.model_id.encode("utf-8"))
        digest.update(_safe_file_fingerprint(self.model_id).encode("utf-8"))
        if self.model_config is not None:
            digest.update(_safe_file_fingerprint(self.model_config).encode("utf-8"))
        if self.original_config_file is not None:
            digest.update(_safe_file_fingerprint(self.original_config_file).encode("utf-8"))
        digest.update(str(self.vae_model_id or "").encode("utf-8"))
        if self.vae_model_id is not None:
            digest.update(_safe_file_fingerprint(self.vae_model_id).encode("utf-8"))
        digest.update(str(self.dtype).encode("utf-8"))
        digest.update(str(self.device).encode("utf-8"))
        if self.device.type == "cuda":
            digest.update(torch.cuda.get_device_name(self.device).encode("utf-8"))
            digest.update(repr(torch.cuda.get_device_capability(self.device)).encode("utf-8"))
        cache_provider_options = {
            key: value
            for key, value in self.trt_provider_options.items()
            if key not in {"trt_detailed_build_log", "trt_dump_subgraphs"}
        }
        if cache_provider_options.get("trt_cuda_graph_enable") and "trt_max_workspace_size" not in cache_provider_options:
            cache_provider_options["trt_max_workspace_size"] = _CUDA_GRAPH_TRT_WORKSPACE_SIZE
        digest.update(
            repr(sorted((str(key), repr(value)) for key, value in cache_provider_options.items())).encode("utf-8")
        )
        digest.update(str(self.hyper_sd).encode("utf-8"))
        digest.update(str(self.cache_tag or "").encode("utf-8"))
        for lora in self.loras:
            digest.update(_safe_file_fingerprint(lora.path).encode("utf-8"))
            digest.update(repr((lora.scale, lora.adapter_name, lora.weight_name)).encode("utf-8"))
        return digest.hexdigest()[:20]

    def _trt_paths(self) -> tuple[Path, Path, Path]:
        cache_root = self.engine_dir / f"{self._cache_key()}"
        return cache_root, cache_root / "onnx", cache_root / "diffusers_source"

    @staticmethod
    def _onnx_cache_ready(path: Path) -> bool:
        if not (path / "model_index.json").is_file():
            return False
        required = (
            path / "unet" / "model.onnx",
            path / "text_encoder" / "model.onnx",
            path / "vae_encoder" / "model.onnx",
            path / "vae_decoder" / "model.onnx",
        )
        return all(item.is_file() for item in required) and any(path.rglob("*.onnx"))

    def _make_trt_provider_options(self, cache_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
        engine_cache = cache_root / "trt_engines"
        engine_cache.mkdir(parents=True, exist_ok=True)
        device_id = _device_index(self.device)
        trt_workspace_size = self._trt_workspace_size()
        if self.trt_provider_options.get("trt_cuda_graph_enable", False):
            trt_workspace_size = min(trt_workspace_size, _CUDA_GRAPH_TRT_WORKSPACE_SIZE)

        trt_options: dict[str, Any] = {
            "device_id": device_id,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(engine_cache),
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(cache_root / "timing.cache"),
            "trt_max_workspace_size": trt_workspace_size,
            "trt_builder_optimization_level": 2,
            "trt_auxiliary_streams": 0,
        }
        trt_options.update(self.trt_provider_options)
        trt_options.pop("trt_build_heuristics_enable", None)

        cuda_options: dict[str, Any] = {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        return providers, [trt_options, cuda_options, {}]

    @staticmethod
    def _prebuild_tensorrt_vae_decoder_engine(
        onnx_dir: Path,
        providers: list[str],
        provider_options: list[dict[str, Any]],
    ) -> None:
        vae_path = onnx_dir / "vae_decoder" / "model.onnx"
        if not vae_path.is_file():
            return

        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:              raise RuntimeError("NumPy and ONNX Runtime are required for VAE graph prebuild.") from exc

        LOGGER.info("Prebuilding the TensorRT VAE decoder engine from %s.", vae_path)
        session = None
        try:
            session = ort.InferenceSession(
                str(vae_path),
                providers=list(providers),
                provider_options=[dict(item) for item in provider_options],
            )
            input_meta = session.get_inputs()[0]
            input_dtype = np.float16 if "float16" in input_meta.type else np.float32
            latent = np.zeros((1, 4, 64, 64), dtype=input_dtype)
            session.run(None, {input_meta.name: latent})
        finally:
            session = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    @staticmethod
    def _prebuild_tensorrt_vae_encoder_engine(
        onnx_dir: Path,
        providers: list[str],
        provider_options: list[dict[str, Any]],
    ) -> None:
        vae_path = onnx_dir / "vae_encoder" / "model.onnx"
        if not vae_path.is_file():
            return

        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:              raise RuntimeError("NumPy and ONNX Runtime are required for VAE graph prebuild.") from exc

        LOGGER.info("Prebuilding the TensorRT VAE encoder engine from %s.", vae_path)
        session = None
        try:
            session = ort.InferenceSession(
                str(vae_path),
                providers=list(providers),
                provider_options=[dict(item) for item in provider_options],
            )
            input_meta = session.get_inputs()[0]
            input_dtype = np.float16 if "float16" in input_meta.type else np.float32
            image = np.zeros((1, 3, 512, 512), dtype=input_dtype)
            session.run(None, {input_meta.name: image})
        finally:
            session = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _load_tensorrt_pipeline(self) -> Any:
        if self.device.type != "cuda":
            raise RuntimeError("TensorRT requires a CUDA device.")
        if self.loras and not self.fuse_loras:
            raise ValueError(
                "TensorRT cannot switch unfused Diffusers LoRAs at runtime. "
                "Use fuse_loras=True or backend='torch'."
            )

        try:
            from optimum.onnxruntime import ORTStableDiffusionPipeline
        except ImportError as exc:              raise RuntimeError("Install optimum-onnx, onnxruntime-gpu, and TensorRT for the TensorRT backend.") from exc

        cache_root, onnx_dir, diffusers_source = self._trt_paths()
        cache_root.mkdir(parents=True, exist_ok=True)
        providers, provider_options = self._make_trt_provider_options(cache_root)

        ort_kwargs = dict(self.model_kwargs)
        ort_kwargs["torch_dtype"] = self.dtype
        ort_kwargs.pop("use_safetensors", None)
        if not self._onnx_cache_ready(onnx_dir):
            source: str
            if self.loras or _is_single_file_source(self.model_id) or self.hyper_sd or self.vae_model_id:
                with _quiet_onnx_export_warnings(), _tiny_vae_export_compatibility(
                    enabled=self.vae_model_id is not None
                ):
                    native_pipe = self._load_native_pipeline(with_loras=True, for_export=True)
                    native_pipe.save_pretrained(diffusers_source, safe_serialization=True)
                    del native_pipe
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                source = str(diffusers_source)
            else:
                source = self.model_id

            LOGGER.info("Exporting the Diffusers pipeline to ONNX for TensorRT.")
            with _quiet_onnx_export_warnings(), _tiny_vae_export_compatibility(
                enabled=self.vae_model_id is not None
            ):
                ort_pipe = ORTStableDiffusionPipeline.from_pretrained(
                    source,
                    export=True,
                    provider="TensorrtExecutionProvider",
                    providers=providers,
                    provider_options=provider_options,
                    use_io_binding=True,
                    **ort_kwargs,
                )
                ort_pipe.save_pretrained(onnx_dir)
        else:
            LOGGER.info("Loading cached ONNX/TensorRT artifacts from %s.", onnx_dir)

        self._prebuild_tensorrt_vae_encoder_engine(onnx_dir, providers, provider_options)
        self._prebuild_tensorrt_vae_decoder_engine(onnx_dir, providers, provider_options)

        ort_pipe = ORTStableDiffusionPipeline.from_pretrained(
            str(onnx_dir),
            export=False,
            provider="TensorrtExecutionProvider",
            providers=providers,
            provider_options=provider_options,
            use_io_binding=True,
        )
        ort_pipe.to(self.device)
        self._prepare_ort_text_encoder_compatibility(ort_pipe)
        if self.hyper_sd:
            self._configure_hyper_sd_scheduler(ort_pipe)
        if hasattr(ort_pipe, "set_progress_bar_config"):
            ort_pipe.set_progress_bar_config(disable=not self.show_progress)
        return ort_pipe

    @staticmethod
    def _prepare_ort_text_encoder_compatibility(pipe: Any) -> None:
        text_encoder = getattr(pipe, "text_encoder", None)
        config = getattr(text_encoder, "config", None)
        if text_encoder is None or config is None:
            return

        try:
            config_values = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        except (AttributeError, TypeError, ValueError):
            return

        for name in ("num_hidden_layers", "num_decoder_layers"):
            if name in text_encoder.__dict__ or name not in config_values:
                continue
            try:
                setattr(text_encoder, name, config_values[name])
            except (AttributeError, TypeError):
                pass

    def _release_pipeline(self) -> None:
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_vae_mode(self, use_tiny_vae: bool) -> None:
        target_vae_model_id = self.tiny_vae_model_id if use_tiny_vae else None
        if self.vae_model_id == target_vae_model_id:
            return

        with self._lock:
            if self.vae_model_id == target_vae_model_id:
                return

            previous_vae_model_id = self.vae_model_id
            previous_backend = self._backend
            self._release_pipeline()
            self.vae_model_id = target_vae_model_id
            self._backend = ""
            self._lora_fused = False
            self._active_lora_names = ()

            try:
                if previous_backend == "torch":
                    self._pipeline = self._load_native_pipeline(with_loras=True, for_export=False)
                    self._backend = "torch"
                else:
                    self._load()
            except Exception as exc:
                self._release_pipeline()
                self.vae_model_id = previous_vae_model_id
                try:
                    if previous_backend == "torch":
                        self._pipeline = self._load_native_pipeline(with_loras=True, for_export=False)
                        self._backend = "torch"
                    else:
                        self._load()
                except Exception as restore_exc:
                    raise RuntimeError(
                        "Could not switch VAE and could not restore the previous pipeline."
                    ) from restore_exc
                raise RuntimeError(
                    f"Could not switch to {'Tiny VAE' if use_tiny_vae else 'the checkpoint VAE'}."
                ) from exc

    def _switch_to_native_backend(self) -> None:
        self._release_pipeline()
        self._pipeline = self._load_native_pipeline(with_loras=True, for_export=False)
        self._backend = "torch"

    def set_lora_weights(
        self,
        adapter_names: str | Sequence[str],
        adapter_weights: float | Sequence[float] | None = None,
    ) -> None:
        if self._backend != "torch":
            raise RuntimeError("Dynamic LoRA switching is available only with backend='torch'.")
        if self._lora_fused:
            raise RuntimeError("The configured LoRAs were fused; recreate the generator with fuse_loras=False.")
        with self._lock:
            self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

    @staticmethod
    def _prepare_img2img_image(
        image: Any,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        import numpy as np
        from PIL import Image

        if isinstance(image, (str, os.PathLike)):
            with Image.open(Path(image).expanduser()) as opened:
                pil_image = opened.convert("RGB")
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            array = np.asarray(pil_image, dtype=np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor / 127.5 - 1.0
            return tensor.to(device=device, dtype=dtype)

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            array = np.asarray(pil_image, dtype=np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor / 127.5 - 1.0
            return tensor.to(device=device, dtype=dtype)

        if isinstance(image, torch.Tensor):
            tensor = image.detach()
            if tensor.ndim == 3:
                if tensor.shape[0] == 3:
                    tensor = tensor.unsqueeze(0)
                elif tensor.shape[-1] == 3:
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                else:
                    raise ValueError("A tensor image must have three RGB channels.")
            elif tensor.ndim == 4 and tensor.shape[1] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError("A tensor image must have shape [3,H,W] or [B,3,H,W].")
            tensor = tensor.to(device=device)
            if not tensor.is_floating_point():
                tensor = tensor.float()
            tensor = tensor.to(dtype=dtype)
            if tensor.shape[-2:] != (height, width):
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            minimum = float(tensor.amin().item())
            maximum = float(tensor.amax().item())
            if 0.0 <= minimum and maximum <= 1.0:
                tensor = tensor * 2.0 - 1.0
            elif -1.0 <= minimum and maximum <= 1.0:
                pass
            elif 0.0 <= minimum and maximum <= 255.0:
                tensor = tensor / 127.5 - 1.0
            else:
                raise ValueError("A tensor image must contain values in [0,1], [-1,1], or [0,255].")
            return tensor

        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError("An array image must have shape [H,W,3] or [H,W,4].")
        if array.shape[-1] == 4:
            array = array[:, :, :3]
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = np.clip(array * 255.0, 0.0, 255.0)
        else:
            array = np.clip(array, 0.0, 255.0)
        pil_image = Image.fromarray(array.astype(np.uint8), mode="RGB")
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(pil_image, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 127.5 - 1.0
        return tensor.to(device=device, dtype=dtype)

    def _encode_img2img_image(
        self,
        image: Any,
        *,
        height: int,
        width: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        image_tensor = self._prepare_img2img_image(
        image,
        height=height,
        width=width,
        device=self.device,
        dtype=self.dtype,
    )
        encoded = self.pipeline.vae.encode(image_tensor, return_dict=True)

        latents = getattr(encoded, "latents", None)
        if latents is None:
            latent_dist = getattr(encoded, "latent_dist", None)
            if latent_dist is None:
                raise RuntimeError("The active VAE encoder returned neither latents nor latent_dist.")

            parameters = getattr(latent_dist, "parameters", None)
            if (
                self.vae_model_id is not None
                and isinstance(parameters, torch.Tensor)
                and parameters.ndim == 4
                and int(parameters.shape[1]) == 4
            ):
                latents = parameters
            else:
                latents = latent_dist.sample(generator=generator)

        scaling_factor = float(getattr(getattr(self.pipeline.vae, "config", None), "scaling_factor", 1.0))
        latents = latents * scaling_factor
        if latents.ndim != 4 or int(latents.shape[1]) != 4:
            raise RuntimeError(f"The VAE encoder returned an invalid latent shape: {tuple(latents.shape)}")
        return latents.to(device=self.device, dtype=self.dtype)

    def _prepare_one_step_img2img_latents(
        self,
        image: Any,
        *,
        height: int,
        width: int,
        strength: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, int]:
        latents = self._encode_img2img_image(
            image,
            height=height,
            width=width,
            generator=generator,
        )
        scheduler = self.pipeline.scheduler
        train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        requested_timestep = int(round((train_timesteps - 1) * strength))
        requested_timestep = max(0, min(requested_timestep, train_timesteps - 1))

        scheduler.set_timesteps(timesteps=[requested_timestep], device=self.device)
        timestep = scheduler.timesteps[:1]
        timestep_batch = timestep.repeat(latents.shape[0])
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=self.device,
            dtype=latents.dtype,
        )
        noisy_latents = scheduler.add_noise(latents, noise, timestep_batch)
        return noisy_latents, int(timestep[0].item())

    def generate_img2img(
        self,
        prompt: str | list[str],
        image: Any,
        *,
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 512,
        strength: float = 0.15,
        num_inference_steps: int = 1,
        guidance_scale: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        output_path: str | os.PathLike[str] | None = None,
        use_tiny_vae: bool = True,
        **pipeline_kwargs: Any,
    ) -> Any:
        self._validate_size(height, width)
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be between 0.0 and 1.0.")
        if num_inference_steps != 1:
            raise ValueError("generate_img2img currently supports exactly one inference step.")
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both.")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator for one-image img2img.")

        self._ensure_vae_mode(use_tiny_vae)
        self._configure_native_request_memory(
            height=height,
            width=width,
            num_images_per_prompt=1,
        )
        if guidance_scale is None:
            guidance_scale = 1.0 if self.hyper_sd else 7.5
        pipeline_guidance_scale = self._pipeline_guidance_scale(guidance_scale)
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        with self._lock, torch.inference_mode():
            latents, timestep = self._prepare_one_step_img2img_latents(
                image,
                height=height,
                width=width,
                strength=float(strength),
                generator=generator,
            )
            kwargs: dict[str, Any] = {
                **pipeline_kwargs,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": 1,
                "timesteps": [timestep],
                "guidance_scale": pipeline_guidance_scale,
                "generator": generator,
                "latents": latents,
                "height": height,
                "width": width,
                "output_type": "pil",
            }
            result = self.pipeline(**kwargs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        images = result.images
        if output_path is not None:
            self._save_images(images, output_path)
        return images[0] if len(images) == 1 else images

    img2img = generate_img2img

    def warmup(
        self,
        *,
        prompt: str = "warmup",
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 1,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        use_tiny_vae: bool = True,
    ) -> float:
        self._validate_size(height, width)
        self._ensure_vae_mode(use_tiny_vae)
        if height is not None and width is not None:
            self._configure_native_request_memory(
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
            )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1.")
        if guidance_scale is None:
                                    guidance_scale = 1.0 if self.hyper_sd else 7.5
        pipeline_guidance_scale = self._pipeline_guidance_scale(guidance_scale)

        start = time.perf_counter()
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": pipeline_guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "output_type": "pil",
        }

        def run_warmup() -> None:
            with self._lock, torch.inference_mode():
                self.pipeline(**kwargs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

        try:
            run_warmup()
        except Exception as exc:
            if self._backend != "tensorrt" or self.requested_backend != "auto":
                raise
            warnings.warn(
                f"TensorRT warm-up failed ({exc!r}); switching to Diffusers CUDA.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._switch_to_native_backend()
            run_warmup()
        elapsed = time.perf_counter() - start
        LOGGER.info("Warm-up finished in %.2fs (backend=%s).", elapsed, self.backend)
        return elapsed

    def generate(
        self,
        prompt: str | list[str],
        *,
        negative_prompt: str | list[str] | None = None,
        height: int | None = 512,
        width: int | None = 512,
        num_inference_steps: int = 30,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        seed: int | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_path: str | os.PathLike[str] | None = None,
        use_tiny_vae: bool = True,
        **pipeline_kwargs: Any,
    ) -> Any:
        if height is not None and width is not None:
            self._validate_size(height, width)
        self._ensure_vae_mode(use_tiny_vae)
        self._configure_native_request_memory(
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
        )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1.")
        if guidance_scale is None:
                        guidance_scale = 1.0 if self.hyper_sd else 7.5
        pipeline_guidance_scale = self._pipeline_guidance_scale(guidance_scale)
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both.")
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": pipeline_guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator,
            **pipeline_kwargs,
        }
        if height is not None:
            kwargs["height"] = height
        if width is not None:
            kwargs["width"] = width

        with self._lock, torch.inference_mode():
            result = self.pipeline(**kwargs)
        images = result.images
        if output_path is not None:
            self._save_images(images, output_path)
        return images[0] if len(images) == 1 else images

    __call__ = generate

    @staticmethod
    def _pipeline_guidance_scale(guidance_scale: float) -> float:
        value = float(guidance_scale)
        if value == 1.0:
            return math.nextafter(1.0, math.inf)
        return value

    def memory_stats(self) -> dict[str, int | float | str]:
        if self.device.type != "cuda":
            return {"device": str(self.device)}
        free_bytes, total_bytes = self._vram_info()
        return {
            "device": str(self.device),
            "backend": self.backend,
            "memory_tier": self._memory_tier(),
            "total_vram_bytes": total_bytes,
            "free_vram_bytes": free_bytes,
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            "trt_workspace_bytes": self._trt_workspace_size(),
        }

    @staticmethod
    def _validate_size(height: int, width: int) -> None:
        if height <= 0 or width <= 0 or height % 8 != 0 or width % 8 != 0:
            raise ValueError("height and width must be positive multiples of 8.")

    @staticmethod
    def _save_images(images: Sequence[Any], output_path: str | os.PathLike[str]) -> None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(images) == 1:
            images[0].save(path)
            return

        suffix = path.suffix or ".png"
        stem = path.with_suffix("")
        for index, image in enumerate(images):
            image.save(stem.parent / f"{stem.name}_{index:03d}{suffix}")