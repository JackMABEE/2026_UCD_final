"""Two-pass PnP diffusion pipeline for Phase 1 texture transfer.

Orchestrates the full ref-pass / gen-pass denoising loop described in
Tumanyan et al., CVPR 2023 §4:

  For each DDIM step t:
    1. Ref pass  — run UNet on z_ref with the source prompt;
                   AttentionInjector captures f^4_t and {A^l_t}.
    2. Gen  pass — run UNet on z_gen with the target prompt;
                   AttentionInjector injects captured features (gated by τ_f / τ_A).

All model components are injected — this class never loads weights itself.
device / dtype come from utils/device.py; τ values come from OmegaConf config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from attn_texture.core.attention_injection import AttentionInjector
from attn_texture.utils.device import get_device_and_dtype, safe_to
from attn_texture.utils.io import pil_to_tensor, tensor_to_pil


class TwoPassPipeline:
    """PnP two-pass texture-transfer pipeline.

    Args:
        unet:         UNet2DConditionModel (or compatible mock) with up_blocks decoder.
        vae:          VAE with .encode() / .decode() and .config.scaling_factor.
        scheduler:    DDIM scheduler with .set_timesteps(), .step(), .add_noise(),
                      .timesteps, and .config.num_train_timesteps.
        tokenizer:    CLIP tokenizer with .model_max_length and .__call__().
        text_encoder: Text encoder nn.Module whose forward returns an object with
                      .last_hidden_state of shape (B, S, D).
        cfg:          OmegaConf node that must contain tau_f and tau_A.
    """

    def __init__(
        self,
        unet: nn.Module,
        vae,
        scheduler,
        tokenizer,
        text_encoder: nn.Module,
        cfg: DictConfig,
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        self._device, self._dtype = get_device_and_dtype()
        self._vae_scale: float = float(vae.config.scaling_factor)

        self.guidance_scale: float = float(cfg.guidance_scale)

        self.injector = AttentionInjector.from_config(unet, cfg)
        self.injector.register_hooks()

        logger.debug(
            "TwoPassPipeline: device={} dtype={} tau_f={} tau_A={} guidance_scale={}",
            self._device,
            self._dtype,
            self.injector.tau_f,
            self.injector.tau_A,
            self.guidance_scale,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        source_image: Image.Image,
        ref_prompt: str,
        gen_prompt: str,
        num_inference_steps: int = 50,
    ) -> Image.Image:
        """Run the two-pass PnP pipeline and return the translated image.

        Args:
            source_image:        RGB PIL Image — the texture reference.
            ref_prompt:          Text prompt describing the source image.
            gen_prompt:          Text prompt describing the desired output.
            num_inference_steps: Number of DDIM denoising steps (default 50).

        Returns:
            RGB PIL Image — the generated texture-transferred result.
        """
        device, dtype = self._device, self._dtype

        with torch.inference_mode():
            # 1. Encode source image to latent z_0  (B, 4, H/8, W/8)
            pixel_tensor = pil_to_tensor(source_image).unsqueeze(0)  # (1, 3, H, W) float32 [0,1]
            pixel_tensor = safe_to(pixel_tensor, device, dtype) * 2.0 - 1.0  # remap to [-1, 1]
            z_0 = self.vae.encode(pixel_tensor).latent_dist.sample() * self._vae_scale

            # 2. Encode prompts to text embeddings  (1, S, D)
            uncond_embeds = self._encode_prompt("")
            ref_embeds = self._encode_prompt(ref_prompt)
            gen_embeds = self._encode_prompt(gen_prompt)

            # 3. Initialise noisy latents z_T from the VAE latent
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            total_steps: int = int(self.scheduler.config.num_train_timesteps)

            noise = torch.randn_like(z_0)
            z_T = self.scheduler.add_noise(z_0, noise, timesteps[:1])
            ref_latents = z_T.clone()
            gen_latents = z_T.clone()

            logger.debug(
                "TwoPassPipeline: starting {} denoising steps (total_train={})",
                num_inference_steps,
                total_steps,
            )

            # 4. Denoising loop
            for t in timesteps:
                step = int(t.item())

                # --- Ref pass: capture features (cond-only; hooks fire here) ---
                self.injector.begin_ref_pass()
                ref_noise_pred = self.unet(
                    ref_latents, t, encoder_hidden_states=ref_embeds, return_dict=True
                ).sample
                self.injector.end_ref_pass()
                ref_latents = self.scheduler.step(ref_noise_pred, t, ref_latents).prev_sample

                # --- Gen pass uncond: hooks idle so captured features stay clean ---
                gen_uncond_pred = self.unet(
                    gen_latents, t, encoder_hidden_states=uncond_embeds, return_dict=True
                ).sample

                # --- Gen pass cond: inject (gated by τ_f / τ_A) ---
                self.injector.begin_gen_pass(step, total_steps)
                gen_cond_pred = self.unet(
                    gen_latents, t, encoder_hidden_states=gen_embeds, return_dict=True
                ).sample
                self.injector.end_gen_pass()

                # CFG formula  # §2 Eq.(1) of Ho et al. 2022
                gen_noise_pred = gen_uncond_pred + self.guidance_scale * (
                    gen_cond_pred - gen_uncond_pred
                )
                gen_latents = self.scheduler.step(gen_noise_pred, t, gen_latents).prev_sample

            # 5. Decode gen latent → PIL image
            decoded = self.vae.decode(gen_latents / self._vae_scale).sample  # (1, 3, H, W)

        return tensor_to_pil(decoded[0].float())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize *prompt* and return text encoder hidden states.

        Args:
            prompt: natural-language description string.

        Returns:
            Tensor of shape (1, S, D) on self._device.
        """
        token_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids.to(self._device)

        with torch.no_grad():
            return self.text_encoder(token_ids).last_hidden_state
