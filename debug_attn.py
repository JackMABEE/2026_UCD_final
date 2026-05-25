import diffusers
print("diffusers", diffusers.__version__)
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
)
unet = pipe.unet
procs = unet.attn_processors
print("attn_processors count:", len(procs))
keys = list(procs.keys())
print("first 6:", keys[:6])
print("last 6:", keys[-6:])
self_attn = [k for k in keys if "attn1" in k]
cross_attn = [k for k in keys if "attn2" in k]
print("self-attn (attn1) count:", len(self_attn))
print("cross-attn (attn2) count:", len(cross_attn))
print("processor type:", type(list(procs.values())[0]) if procs else "EMPTY")
