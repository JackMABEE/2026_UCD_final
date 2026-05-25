import inspect
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# Show set_attn_processor
src = inspect.getsource(UNet2DConditionModel.set_attn_processor)
print("=== set_attn_processor ===")
print(src[:3000])
print()

# Show attn_processors property
src2 = inspect.getsource(UNet2DConditionModel.attn_processors.fget)
print("=== attn_processors ===")
print(src2[:2000])
