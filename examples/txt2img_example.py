from pathlib import Path
from fast_stable_diffusion import FastStableDiffusion, LoRA

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sd" / "v1-5-pruned-emaonly.safetensors"
FAST_LORA_PATH = BASE_DIR / "models" / "lora" / "Hyper-SD15-2steps-lora.safetensors"
#CUSTOM_LORA_PATH = BASE_DIR / "models" / "lora" / "some_lora.safetensors"
TINY_VAE_PATH = BASE_DIR / "models" / "vae"
ENGINE_DIR = BASE_DIR / ".sd_trt_cache"
OUTPUT_PATH = BASE_DIR / "txt2img_output.png"
PROMPT = "a photo of an astronaut riding a horse on mars"

IMAGE_SIZE = 512
SEED = 42

print("Preparing ONNX/TensorRT engines, including the fast VAE encoder...")

#Load model
generator = FastStableDiffusion(
    model_id=MODEL_PATH,
    backend="auto",
    loras=[LoRA(FAST_LORA_PATH)], #LoRA(CUSTOM_LORA_PATH)
    hyper_sd=True,
    vae_model_id=TINY_VAE_PATH,
    engine_dir=ENGINE_DIR,
    disable_safety_checker=True,
)

#Model warmup
generator.warmup(
    prompt=PROMPT,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE,
    num_inference_steps=1,
    guidance_scale=1.0,
    use_tiny_vae=True,
)

#Generate image
print("Generating one-step txt2img...")
generator.generate(
    prompt=PROMPT,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE,
    num_inference_steps=1,
    guidance_scale=1.0,
    seed=SEED,
    use_tiny_vae=True,
    output_path=OUTPUT_PATH, #Returns PIL if no path provided
)
