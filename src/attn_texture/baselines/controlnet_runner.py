"""ControlNet-conditioned image-to-image baseline runner.

Implements txt2img inference conditioned on Canny edges (Canny et al. 1986;
ControlNet: Zhang et al., ICCV 2023):
  1. Canny-preprocess source_image → edge condition tensor.
  2. Initialise latents from torch.randn (txt2img — source is never VAE-encoded).
  3. At each DDIM step:
       down_res, mid_res = controlnet(latents, t, text_embeds, controlnet_cond=cond)
       noise_pred = unet(latents, t, text_embeds,
                         down_block_additional_residuals=down_res,
                         mid_block_additional_residual=mid_res)
       latents = scheduler.step(noise_pred, t, latents).prev_sample
  4. Decode final latents via VAE → RGB PIL Image.

Canny thresholds (canny_low, canny_high) are read from OmegaConf config —
never hardcoded (CLAUDE.md §6 rule 1).  Like SDEditRunner, model objects live
in a factory closure so with_isolated_model can free GPU/MPS memory after
each run() call (CLAUDE.md §5.2).
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from attn_texture.utils.device import get_device_and_dtype, safe_to
from attn_texture.utils.io import pil_to_tensor, tensor_to_pil
from attn_texture.utils.memory import with_isolated_model


class _Components:
    """Bundle of injected model objects used inside run()."""

    __slots__ = ("unet", "vae", "scheduler", "tokenizer", "text_encoder", "controlnet")

    def __init__(self, unet, vae, scheduler, tokenizer, text_encoder, controlnet) -> None:
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.controlnet = controlnet


class ControlNetRunner:
    """ControlNet-conditioned denoising baseline.

    Args:
        unet:         UNet2DConditionModel accepting down_block_additional_residuals
                      and mid_block_additional_residual kwargs.
        vae:          VAE with .decode() and .config.scaling_factor.
        scheduler:    DDIM scheduler with .set_timesteps(), .step(), .timesteps.
        tokenizer:    CLIP tokenizer with .model_max_length and .__call__().
        text_encoder: nn.Module whose forward returns an object with .last_hidden_state.
        controlnet:   ControlNetModel whose forward returns
                      (list[down_res], mid_res) when return_dict=False.
        cfg:          OmegaConf node; must contain canny_low and canny_high.

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
        controlnet: nn.Module,
        cfg: DictConfig,
    ) -> None:
        # Read thresholds early — raises ConfigAttributeError if absent (CLAUDE.md §6)
        self._canny_low: int = int(cfg.canny_low)
        self._canny_high: int = int(cfg.canny_high)
        self._guidance_scale: float = float(cfg.guidance_scale)

        self._factory: Callable[[], _Components] = lambda: _Components(
            unet, vae, scheduler, tokenizer, text_encoder, controlnet
        )
        self._vae_scale: float = float(vae.config.scaling_factor)
        self._device, self._dtype = get_device_and_dtype()

        logger.debug(
            "ControlNetRunner: device={} dtype={} canny=({}, {}) guidance_scale={}",
            self._device,
            self._dtype,
            self._canny_low,
            self._canny_high,
            self._guidance_scale,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        source_image: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
    ) -> Image.Image:
        """Run ControlNet-conditioned txt2img and return the generated image.

        Args:
            source_image:        RGB PIL Image used only for Canny edge extraction.
            prompt:              Text prompt describing the desired output.
            num_inference_steps: Number of DDIM denoising steps (default 50).

        Returns:
            RGB PIL Image.
        """
        with with_isolated_model(self._factory) as comps:
            return self._run_inner(comps, source_image, prompt, num_inference_steps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inner(
        self,
        comps: _Components,
        source_image: Image.Image,
        prompt: str,
        num_inference_steps: int,
    ) -> Image.Image:
        device, dtype = self._device, self._dtype

        # 1. Canny-preprocess → condition tensor (1, 3, H, W) in [0, 1]  (CPU-only, no grad)
        controlnet_cond = self._canny_condition(source_image, device, dtype)

        with torch.inference_mode():
            # 2. Initialise random latents — spatial dims derived from source image
            W_img, H_img = source_image.size
            latents = torch.randn(
                1, 4, H_img // 8, W_img // 8,
                device=device,
                dtype=dtype,
            )

            # 3. Encode target prompt and unconditional embedding → (1, S, D) each
            uncond_embeds = self._encode_prompt(comps, "")
            text_embeds = self._encode_prompt(comps, prompt)
            # Batch for a single forward per step  (B=2, S, D)
            batched_embeds = torch.cat([uncond_embeds, text_embeds])
            # Duplicate condition for batched controlnet forward  (B=2, 3, H, W)
            batched_cond = torch.cat([controlnet_cond, controlnet_cond])

            # 4. Build timestep schedule
            comps.scheduler.set_timesteps(num_inference_steps)

            logger.debug(
                "ControlNetRunner: {} steps, latent shape {}",
                num_inference_steps,
                tuple(latents.shape),
            )

            # 5. Denoising loop with CFG
            for t in comps.scheduler.timesteps:
                batched_latents = torch.cat([latents, latents])
                down_res, mid_res = comps.controlnet(
                    batched_latents,
                    t,
                    encoder_hidden_states=batched_embeds,
                    controlnet_cond=batched_cond,
                    return_dict=False,
                )

                batched_pred = comps.unet(
                    batched_latents,
                    t,
                    encoder_hidden_states=batched_embeds,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                    return_dict=True,
                ).sample

                uncond_pred, cond_pred = batched_pred.chunk(2)
                noise_pred = uncond_pred + self._guidance_scale * (cond_pred - uncond_pred)
                latents = comps.scheduler.step(noise_pred, t, latents).prev_sample

            # 6. Decode → PIL
            decoded = comps.vae.decode(latents / self._vae_scale).sample  # (1, 3, H, W) in ~[-1, 1]
            decoded = decoded / 2.0 + 0.5  # remap to [0, 1] — matches VaeImageProcessor.postprocess
        return tensor_to_pil(decoded[0].float())

    def _canny_condition(
        self,
        image: Image.Image,
        device: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Canny-edge-detect *image* and return a (1, 3, H, W) float tensor in [0, 1].

        Args:
            image:  RGB PIL source image.
            device: target device string.
            dtype:  target dtype.

        Returns:
            Tensor of shape (1, 3, H, W) with edge map replicated across channels.
        """
        arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self._canny_low, self._canny_high)  # (H, W) uint8
        edges_rgb = np.stack([edges, edges, edges], axis=-1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(edges_rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return safe_to(tensor, device, dtype)

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
