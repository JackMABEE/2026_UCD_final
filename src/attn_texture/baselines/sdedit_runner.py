"""SDEdit image-to-image baseline runner.

Implements the SDEdit algorithm (Meng et al., ICLR 2022):
  1. Encode source image to latent z_0 via VAE.
  2. Add noise up to timestep t_s determined by strength ∈ (0, 1].
  3. Denoise from t_s → 0 with the target prompt using DDIM.
  4. Decode the denoised latent → PIL image.

Components are never held as persistent attributes; every call to run()
acquires them through with_isolated_model so GPU / MPS memory is freed
when the call returns (CLAUDE.md §5.2).
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from attn_texture.utils.device import get_device_and_dtype, safe_to
from attn_texture.utils.io import pil_to_tensor, tensor_to_pil
from attn_texture.utils.memory import with_isolated_model


class _Components:
    """Lightweight bundle of injected model objects used inside run()."""

    __slots__ = ("unet", "vae", "scheduler", "tokenizer", "text_encoder")

    def __init__(self, unet, vae, scheduler, tokenizer, text_encoder) -> None:
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder


class SDEditRunner:
    """SDEdit baseline runner.

    Args:
        unet:         UNet2DConditionModel (or compatible mock).
        vae:          VAE with .encode() / .decode() and .config.scaling_factor.
        scheduler:    DDIM scheduler with .set_timesteps(), .step(), .add_noise(),
                      .timesteps, and .config.num_train_timesteps.
        tokenizer:    CLIP tokenizer with .model_max_length and .__call__().
        text_encoder: nn.Module whose forward returns an object with .last_hidden_state.
        cfg:          OmegaConf node (reserved for future hyperparameters).

    Note:
        Model objects are captured in a factory closure, not stored as direct
        attributes, so with_isolated_model can free them after each run() call.
    """

    def __init__(
        self,
        unet: nn.Module,
        vae: Any,
        scheduler: Any,
        tokenizer: Any,
        text_encoder: nn.Module,
        cfg: DictConfig,
    ) -> None:
        self._factory: Callable[[], _Components] = lambda: _Components(
            unet, vae, scheduler, tokenizer, text_encoder
        )
        self._vae_scale: float = float(vae.config.scaling_factor)
        self._guidance_scale: float = float(cfg.guidance_scale)
        self._device, self._dtype = get_device_and_dtype()

        logger.debug(
            "SDEditRunner: device={} dtype={} vae_scale={} guidance_scale={}",
            self._device,
            self._dtype,
            self._vae_scale,
            self._guidance_scale,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        source_image: Image.Image,
        prompt: str,
        strength: float = 0.8,
        num_inference_steps: int = 50,
    ) -> Image.Image:
        """Run SDEdit and return the translated image.

        Args:
            source_image:        RGB PIL Image — the texture source.
            prompt:              Text prompt describing the desired output.
            strength:            Noise strength in [0.0, 1.0].
                                 0.0 → no denoising (return VAE round-trip of source).
                                 1.0 → denoise from pure noise (full creative freedom).
            num_inference_steps: Total DDIM steps used to build the timestep schedule.

        Returns:
            RGB PIL Image.
        """
        with with_isolated_model(self._factory) as comps:
            return self._run_inner(comps, source_image, prompt, strength, num_inference_steps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inner(
        self,
        comps: _Components,
        source_image: Image.Image,
        prompt: str,
        strength: float,
        num_inference_steps: int,
    ) -> Image.Image:
        device, dtype = self._device, self._dtype

        with torch.inference_mode():
            # 1. Encode source image → z_0  (B, 4, H/8, W/8)
            pixel_tensor = pil_to_tensor(source_image).unsqueeze(0)  # (1, 3, H, W) [0, 1]
            pixel_tensor = safe_to(pixel_tensor, device, dtype) * 2.0 - 1.0  # [-1, 1]
            z_0 = comps.vae.encode(pixel_tensor).latent_dist.sample() * self._vae_scale

            # 2. Build full timestep schedule, then slice to the active window
            comps.scheduler.set_timesteps(num_inference_steps)
            timesteps = comps.scheduler.timesteps  # descending, e.g. [999, ..., 0]

            start_step = round(num_inference_steps * (1.0 - strength))
            active_timesteps = timesteps[start_step:]  # may be empty when strength=0.0

            logger.debug(
                "SDEditRunner: strength={} steps={} active_steps={}",
                strength,
                num_inference_steps,
                len(active_timesteps),
            )

            # 3. Add noise up to the entry timestep (skip entirely if no active steps)
            if len(active_timesteps) == 0:
                latents = z_0
            else:
                noise = torch.randn_like(z_0)
                t_start = active_timesteps[:1]  # tensor([t_s])
                latents = comps.scheduler.add_noise(z_0, noise, t_start)

            # 4. Encode target prompt and unconditional embedding → (1, S, D) each
            uncond_embeds = self._encode_prompt(comps, "")
            text_embeds = self._encode_prompt(comps, prompt)
            # Batch together for a single UNet forward per step  (B=2, S, D)
            batched_embeds = torch.cat([uncond_embeds, text_embeds])

            # 5. Denoising loop with CFG
            for t in active_timesteps:
                batched_latents = torch.cat([latents, latents])
                batched_pred = comps.unet(
                    batched_latents, t, encoder_hidden_states=batched_embeds, return_dict=True
                ).sample
                uncond_pred, cond_pred = batched_pred.chunk(2)
                noise_pred = uncond_pred + self._guidance_scale * (cond_pred - uncond_pred)
                latents = comps.scheduler.step(noise_pred, t, latents).prev_sample

        # 6. Decode → PIL
        with torch.inference_mode():
            decoded = comps.vae.decode(latents / self._vae_scale).sample  # (1, 3, H, W)
        return tensor_to_pil(decoded[0].float())

    def _encode_prompt(self, comps: _Components, prompt: str) -> torch.Tensor:
        """Tokenize *prompt* and return text encoder hidden states (1, S, D)."""
        ids = comps.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=comps.tokenizer.model_max_length,
            truncation=True,
        ).input_ids.to(self._device)

        with torch.no_grad():
            return comps.text_encoder(ids).last_hidden_state
