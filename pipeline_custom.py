"""
CustomImg2ImgPipeline
=====================
Wraps ``StableDiffusionImg2ImgPipeline`` and integrates:
  1. Attention capture / injection  (AttentionStore / AttentionInjector)
  2. FFT-based frequency filtering on intermediate latents

The pipeline performs TWO forward passes per call:
  Pass A (reference) — source image, attention K/Q are *captured*.
  Pass B (target)    — source image + text prompt, attention K/Q are *injected*
                       from Pass A; optional frequency filter is applied at each
                       denoising step.

Usage
-----
    pipe = CustomImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.setup_control(cfg["attention_control"], cfg["frequency_filter"])
    result = pipe.run(source_pil, prompt="...", **kwargs)
"""

from __future__ import annotations

import torch
from PIL import Image
from typing import Any, Dict, List, Optional, Union

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import logging

from models.attention_control import AttentionStore, AttentionInjector
from utils.frequency_filter import FrequencyFilter

logger = logging.get_logger(__name__)


class CustomImg2ImgPipeline:
    """
    Thin wrapper around diffusers' StableDiffusionImg2ImgPipeline.

    Parameters
    ----------
    pipeline : StableDiffusionImg2ImgPipeline
        A fully-initialised diffusers pipeline (already on target device).
    """

    def __init__(self, pipeline: StableDiffusionImg2ImgPipeline):
        self.pipe = pipeline
        self.device = pipeline.device
        self.dtype = pipeline.unet.dtype

        # Components — set by setup_control()
        self._store: Optional[AttentionStore] = None
        self._injector: Optional[AttentionInjector] = None
        self._freq_filter: Optional[FrequencyFilter] = None

        # Config flags
        self._attn_enabled: bool = False
        self._freq_enabled: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        torch_dtype=torch.float16,
        device: str = "cuda",
        **kwargs,
    ) -> "CustomImg2ImgPipeline":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, **kwargs
        ).to(device)
        pipe.safety_checker = None  # disable for research use
        return cls(pipe)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def setup_control(
        self,
        attn_cfg: Dict[str, Any],
        freq_cfg: Dict[str, Any],
    ):
        """
        Initialise attention control and frequency filter from config dicts
        (mirrors the YAML structure in configs/experiment_01.yaml).
        """
        # --- Attention ---
        self._attn_enabled = attn_cfg.get("enabled", False)
        if self._attn_enabled:
            target_layers = attn_cfg.get("target_layers", None)
            self._store = AttentionStore(target_layers=target_layers)
            self._injector = AttentionInjector(
                store=self._store,
                injection_steps=attn_cfg.get("injection_steps", list(range(10))),
                injection_mode=attn_cfg.get("injection_mode", "replace"),
                blend_alpha=attn_cfg.get("blend_alpha", 1.0),
            )

        # --- Frequency filter ---
        self._freq_enabled = freq_cfg.get("enabled", False)
        if self._freq_enabled:
            self._freq_filter = FrequencyFilter(
                low_pass_radius=freq_cfg.get("low_pass_radius", 0.25),
                mode=freq_cfg.get("mode", "hybrid"),
                high_freq_weight=freq_cfg.get("high_freq_weight", 0.5),
            )

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self,
        source_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute the two-pass pipeline and return a result dict.

        Returns
        -------
        {
            "image"         : PIL.Image,
            "reference_image": PIL.Image  (pass-A output, for comparison),
            "latents_history": list of Tensor  (one per step, if freq filter active),
        }
        """
        generator = self._make_generator(seed)

        # -------- Pass A: reference (no prompt injection) --------
        if self._attn_enabled and self._store is not None:
            store_handle = self._store.register(self.pipe.unet)

        ref_output = self._run_diffusers(
            source_image=source_image,
            prompt=prompt,          # same prompt — we want structural layout
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=self._make_generator(seed),
            callback=self._make_step_callback(
                store=self._store if self._attn_enabled else None,
                injector=None,
                freq_filter=None,
            ),
        )
        ref_image: Image.Image = ref_output.images[0]

        if self._attn_enabled and self._store is not None:
            store_handle.remove()
            logger.info(
                "Pass A complete. Captured attention from %d layers.",
                len(self._store.store),
            )

        # -------- Pass B: target (prompt + attention injection) --------
        if self._attn_enabled and self._injector is not None:
            inj_handle = self._injector.register(self.pipe.unet)

        latents_history: List[torch.Tensor] = []

        tgt_output = self._run_diffusers(
            source_image=source_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=self._make_step_callback(
                store=None,
                injector=self._injector if self._attn_enabled else None,
                freq_filter=self._freq_filter if self._freq_enabled else None,
                latents_history=latents_history,
            ),
        )
        gen_image: Image.Image = tgt_output.images[0]

        if self._attn_enabled and self._injector is not None:
            inj_handle.remove()

        return {
            "image": gen_image,
            "reference_image": ref_image,
            "latents_history": latents_history,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_diffusers(
        self,
        source_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        strength: float,
        guidance_scale: float,
        generator: torch.Generator,
        callback,
    ):
        # diffusers ≥ 0.21 uses callback_on_step_end; fall back to callback.
        try:
            return self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=source_image,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        except TypeError:
            # Older diffusers API
            return self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=source_image,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=callback,
                callback_steps=1,
            )

    @staticmethod
    def _make_step_callback(
        store: Optional[AttentionStore],
        injector: Optional[AttentionInjector],
        freq_filter: Optional[FrequencyFilter],
        latents_history: Optional[List[torch.Tensor]] = None,
    ):
        """
        Returns a callback compatible with both diffusers callback APIs.
        The callback:
          1. Advances step counters in store / injector.
          2. Optionally applies frequency filter to the current latents.
          3. Optionally records latents in latents_history.
        """

        def callback(pipe_or_step, step_or_ts, timestep_or_latents, callback_kwargs=None):
            # Unified handling for new (callback_on_step_end) and old (callback) APIs.
            if callback_kwargs is not None:
                # New API: callback_on_step_end(pipe, step, timestep, callback_kwargs)
                latents = callback_kwargs.get("latents")
            else:
                # Old API: callback(step, timestep, latents)
                latents = timestep_or_latents

            if store is not None:
                store.step()
            if injector is not None:
                injector.step()

            if freq_filter is not None and latents is not None:
                latents = freq_filter(latents)
                if callback_kwargs is not None:
                    callback_kwargs["latents"] = latents

            if latents_history is not None and latents is not None:
                latents_history.append(latents.detach().cpu())

            if callback_kwargs is not None:
                return callback_kwargs

        return callback

    def _make_generator(self, seed: Optional[int]) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        return gen
