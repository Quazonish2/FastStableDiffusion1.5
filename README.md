# Documentation
Just view/read examples. It's pretty self-explanatory

# Requirements
In requirements folder pick CUDA version you use and install it

# Speed
On my RTX 4050 one image generation takes around 100 ms. You can measure speed for your GPU using scripts inside tests folder.

# Models
Download [Stable diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors) and place it in models/sd.
Download [Hyper SD](https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SD15-2steps-lora.safetensors)(although it says 2 steps, it perfectly works with 1 step) and place it in models/lora.
You can also place your own loras in models/lora, but don't forget to also add it in your code.

# Image of someone's face source
https://www.magnific.com/free-photo/portrait-white-man-isolated_3199590.htm