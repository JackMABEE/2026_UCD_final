"""MasaCtrl mutual self-attention baseline runner.

Implements MasaCtrl (Cao et al., ICCV 2023):
  Mutual Self-Attention Control for Consistent Image Synthesis and Editing.
  https://github.com/TencentARC/MasaCtrl

Algorithm
---------
1. Encode source image via VAE → z_src_0.
2. DDIM inversion: run denoising loop in reverse (ascending timesteps) starting
   from z_src_0 to produce an approximate noisy latent z_T.
3. Register _MasaCtrlAttnProcessor on every UNet self-attention layer.
4. Denoising loop (batch=4):
     [z_src_unc, z_tgt_unc, z_src_cond, z_tgt_cond]
   Active processors inject K and V from source slots (0, 2) into target slots
   (1, 3) so target queries attend over source keys/values.
5. Decode target latent z_tgt_0 via VAE → RGB PIL Image.

K/V injection is gated by two thresholds (§4.2 of the MasaCtrl paper):
  step_idx  >= step_start   — skip the highly-noisy early steps where
                              structural layout has not formed yet.
  layer_idx >= layer_start  — skip shallow encoder layers that encode
                              low-level geometry rather than appearance.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from attn_texture.utils.device import get_device_and_dtype, safe_to
from attn_texture.utils.io import pil_to_tensor, tensor_to_pil
from attn_texture.utils.memory import with_isolated_model


class _MasaCtrlAttnProcessor:
    """Self-attention processor that injects source K/V into the target query stream.

    Expected batch layout (batch=4) during the denoising loop:
      index 0: source + unconditional embedding  (src_unc)
      index 1: target + unconditional embedding  (tgt_unc)
      index 2: source + conditional embedding    (src_cond)
      index 3: target + conditional embedding    (tgt_cond)

    Injection (when both gating thresholds are satisfied):
      k[1] ← k[0],  k[3] ← k[2]   # tgt borrows K from src   (§4.2 Eq. 2)
      v[1] ← v[0],  v[3] ← v[2]   # tgt borrows V from src

    Args:
        layer_idx:   Zero-based position of this layer among all UNet attention layers.
        step_start:  First denoising step index at which injection is active.
        layer_start: First layer index at which injection is active.
    """

    def __init__(self, layer_idx: int, step_start: int, layer_start: int) -> None:
        self.layer_idx = layer_idx
        self.step_start = step_start
        self.layer_start = layer_start
        self._cur_step: int = 0  # updated by MasaCtrlRunner at each denoising step

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run self-attention with optional mutual K/V injection.

        For cross-attention layers (encoder_hidden_states is not None) the processor
        behaves identically to AttnProcessor2_0 — K/V are projected from the text
        embeddings and no injection is applied.  Injection is self-attention only.

        Args:
            attn:                 diffusers Attention module (provides to_q/k/v etc.).
            hidden_states:        (B, S, D) query input.
            encoder_hidden_states: text embeddings for cross-attention; None for self-attn.
            attention_mask:       optional bias passed to get_attention_scores.

        Returns:
            (B, S, D) attended output.
        """
        B, _seq_len, _dim = hidden_states.shape

        # Cross-attention: K/V come from encoder_hidden_states (text embeddings, 768-dim).
        # Self-attention: K/V come from hidden_states (spatial features).
        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_input)
        v = attn.to_v(kv_input)

        # § 4.2 Mutual Self-Attention: inject K/V only for self-attention layers.
        if (
            not is_cross_attention
            and B == 4
            and self._cur_step >= self.step_start
            and self.layer_idx >= self.layer_start
        ):
            k = k.clone()
            v = v.clone()
            k[1] = k[0]  # tgt_unc  borrows K from src_unc
            k[3] = k[2]  # tgt_cond borrows K from src_cond
            v[1] = v[0]
            v[3] = v[2]

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        attn_weights = attn.get_attention_scores(q, k, attention_mask)
        hidden_states = torch.bmm(attn_weights, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class _Components:
    """Lightweight bundle of injected model objects used inside run()."""

    __slots__ = ("unet", "vae", "scheduler", "tokenizer", "text_encoder")

    def __init__(self, unet, vae, scheduler, tokenizer, text_encoder) -> None:
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder


class MasaCtrlRunner:
    """MasaCtrl mutual self-attention baseline runner.

    Args:
        unet:         UNet2DConditionModel with .attn_processors and .set_attn_processor().
        vae:          VAE with .encode() / .decode() and .config.scaling_factor.
        scheduler:    DDIM scheduler with .set_timesteps(), .step(), .timesteps.
        tokenizer:    CLIP tokenizer with .model_max_length and .__call__().
        text_encoder: nn.Module whose forward returns an object with .last_hidden_state.
        cfg:          OmegaConf node; must contain step_start, layer_start,
                      guidance_scale, num_inference_steps.

    Note:
        Model objects are captured in a factory closure, not stored as direct
        attributes, so with_isolated_model can free GPU/MPS memory after each
        run() call (CLAUDE.md §5.2).
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
        self._step_start: int = int(cfg.step_start)
        self._layer_start: int = int(cfg.layer_start)
        self._guidance_scale: float = float(cfg.guidance_scale)

        self._factory: Callable[[], _Components] = lambda: _Components(
            unet, vae, scheduler, tokenizer, text_encoder
        )
        self._vae_scale: float = float(vae.config.scaling_factor)
        self._device, self._dtype = get_device_and_dtype()

        logger.debug(
            "MasaCtrlRunner: device={} dtype={} step_start={} layer_start={} gs={}",
            self._device,
            self._dtype,
            self._step_start,
            self._layer_start,
            self._guidance_scale,
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
        """Run MasaCtrl and return the generated image.

        Args:
            source_image:        RGB PIL Image — the texture/appearance source.
            ref_prompt:          Describes the source content; used during DDIM
                                 inversion so that source latents are accurate.
            gen_prompt:          Describes the desired output.
            num_inference_steps: Total DDIM steps for both inversion and generation.

        Returns:
            RGB PIL Image.
        """
        with with_isolated_model(self._factory) as comps:
            return self._run_inner(comps, source_image, ref_prompt, gen_prompt, num_inference_steps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inner(
        self,
        comps: _Components,
        source_image: Image.Image,
        ref_prompt: str,
        gen_prompt: str,
        num_inference_steps: int,
    ) -> Image.Image:
        device, dtype = self._device, self._dtype

        with torch.inference_mode():
            # 1. Encode source image → z_src_0  (1, 4, H/8, W/8)
            pixel_tensor = pil_to_tensor(source_image).unsqueeze(0)
            pixel_tensor = safe_to(pixel_tensor, device, dtype) * 2.0 - 1.0  # [-1, 1]
            z_src = comps.vae.encode(pixel_tensor).latent_dist.sample() * self._vae_scale

            # 2. Build timestep schedule
            comps.scheduler.set_timesteps(num_inference_steps)
            timesteps = comps.scheduler.timesteps  # descending [T-1, ..., 0]

            # 3. Encode all prompts once
            src_uncond = self._encode_prompt(comps, "")
            src_cond = self._encode_prompt(comps, ref_prompt)
            tgt_uncond = self._encode_prompt(comps, "")
            tgt_cond = self._encode_prompt(comps, gen_prompt)
            src_embeds = torch.cat([src_uncond, src_cond])  # (2, S, D) for inversion CFG

            # 4. DDIM inversion: z_src_0 → z_src_T via approximate reverse ODE
            #    Iterate ascending (reversed timesteps list) — no grad, source path only.
            z_inv = z_src.clone()
            for t in reversed(timesteps):
                inp = torch.cat([z_inv, z_inv])  # (2, 4, H/8, W/8)
                noise_pred = comps.unet(
                    inp, t, encoder_hidden_states=src_embeds, return_dict=True
                ).sample
                unc, cond = noise_pred.chunk(2)
                noise_pred = unc + self._guidance_scale * (cond - unc)
                z_inv = comps.scheduler.step(noise_pred, t, z_inv).prev_sample

            # 5. Register MasaCtrl processors (after inversion, before generation)
            self._register_processors(comps.unet)

            # 6. Both source and target start from the inverted latent z_T
            z_src_t = z_inv.clone()
            z_tgt_t = z_inv.clone()

            # Batch of 4 embeddings: [src_unc, tgt_unc, src_cond, tgt_cond]
            batched_embeds = torch.cat([src_uncond, tgt_uncond, src_cond, tgt_cond])

            # 7. Denoising loop: both paths run in a single batched forward per step
            for step_idx, t in enumerate(timesteps):
                # Inform processors of the current step so gating is correct
                for proc in comps.unet.attn_processors.values():
                    if hasattr(proc, "_cur_step"):
                        proc._cur_step = step_idx

                # Batch = 4: [src_unc, tgt_unc, src_cond, tgt_cond]
                latents_4 = torch.cat([z_src_t, z_tgt_t, z_src_t, z_tgt_t])
                noise_pred_4 = comps.unet(
                    latents_4, t, encoder_hidden_states=batched_embeds, return_dict=True
                ).sample

                pred_src_unc, pred_tgt_unc, pred_src_cond, pred_tgt_cond = noise_pred_4.chunk(4)

                pred_src = pred_src_unc + self._guidance_scale * (pred_src_cond - pred_src_unc)
                pred_tgt = pred_tgt_unc + self._guidance_scale * (pred_tgt_cond - pred_tgt_unc)

                z_src_t = comps.scheduler.step(pred_src, t, z_src_t).prev_sample
                z_tgt_t = comps.scheduler.step(pred_tgt, t, z_tgt_t).prev_sample

        # 8. Decode target latent → PIL
        with torch.inference_mode():
            decoded = comps.vae.decode(z_tgt_t / self._vae_scale).sample  # (1, 3, H, W)
            decoded = decoded / 2.0 + 0.5
        return tensor_to_pil(decoded[0].float())

    def _register_processors(self, unet: nn.Module) -> None:
        """Replace self-attention (attn1) processors with _MasaCtrlAttnProcessor.

        Cross-attention (attn2) processors are preserved unchanged so that text
        conditioning is unaffected.  set_attn_processor requires the full dict
        (all 32 entries for SD 1.5), so we start from the existing processors and
        only overwrite the attn1 entries.
        """
        existing = unet.attn_processors  # full dict, e.g. 32 entries for SD 1.5
        processors = dict(existing)      # copy — we will overwrite attn1 entries only

        self_attn_idx = 0
        for name in existing:
            if "attn1" in name:  # self-attention layers only (§4.2)
                processors[name] = _MasaCtrlAttnProcessor(
                    layer_idx=self_attn_idx,
                    step_start=self._step_start,
                    layer_start=self._layer_start,
                )
                self_attn_idx += 1

        n_total = len(processors)  # capture before set_attn_processor pops entries
        unet.set_attn_processor(processors)
        logger.debug(
            "MasaCtrlRunner: registered {} self-attn processors out of {} total "
            "(step_start={}, layer_start={})",
            self_attn_idx,
            n_total,
            self._step_start,
            self._layer_start,
        )

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
