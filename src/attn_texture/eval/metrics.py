"""Image quality metrics over PIL images.

All functions accept PIL images and return a scalar float.  Used by
eval/shootout.py to compare each baseline/ours output against the reference.

Implementations
---------------
ssim          — scikit-image structural_similarity, channel_axis=2, data_range=255
psnr          — scikit-image peak_signal_noise_ratio, data_range=255
lpips_score   — lpips.LPIPS(net="alex"); singleton loaded once per process
clip_score    — openai/clip-vit-base-patch32 cosine similarity (image ↔ prompt)
dino_distance — facebook/dino-vits8 CLS-token distance (1 − cosine_similarity)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoFeatureExtractor, AutoModel, CLIPModel, CLIPProcessor

import lpips as _lpips_lib

_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_DINO_MODEL_ID = "facebook/dino-vits8"

# Module-level singletons — each model loads once per process.
_lpips_model: "_lpips_lib.LPIPS | None" = None
_clip_model: "CLIPModel | None" = None
_clip_processor: "CLIPProcessor | None" = None
_dino_model: "AutoModel | None" = None
_dino_extractor: "AutoFeatureExtractor | None" = None


def _get_lpips_model() -> "_lpips_lib.LPIPS":
    global _lpips_model
    if _lpips_model is None:
        logger.debug("Loading LPIPS AlexNet weights (first call only)…")
        _lpips_model = _lpips_lib.LPIPS(net="alex", verbose=False)
        _lpips_model.eval()
    return _lpips_model


def _get_clip() -> tuple["CLIPModel", "CLIPProcessor"]:
    global _clip_model, _clip_processor
    if _clip_model is None:
        logger.debug("Loading CLIP {} (first call only)…", _CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_ID)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _get_dino() -> tuple["AutoModel", "AutoFeatureExtractor"]:
    global _dino_model, _dino_extractor
    if _dino_model is None:
        logger.debug("Loading DINO {} (first call only)…", _DINO_MODEL_ID)
        _dino_extractor = AutoFeatureExtractor.from_pretrained(_DINO_MODEL_ID)
        _dino_model = AutoModel.from_pretrained(_DINO_MODEL_ID)
        _dino_model.eval()
    return _dino_model, _dino_extractor


def _to_numpy_rgb(img: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 HWC numpy array."""
    return np.array(img.convert("RGB"))


def _check_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Images must have identical spatial dimensions, "
            f"got {a.shape[:2]} vs {b.shape[:2]}"
        )


def ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    """Structural Similarity Index between two RGB PIL images.

    Args:
        img_a: reference image.
        img_b: comparison image; must have the same spatial dimensions as img_a.

    Returns:
        SSIM score in [-1, 1]; identical images return 1.0.

    Raises:
        ValueError: if the images differ in spatial dimensions.
    """
    a, b = _to_numpy_rgb(img_a), _to_numpy_rgb(img_b)
    _check_shape(a, b)
    score, _ = structural_similarity(a, b, channel_axis=2, data_range=255, full=True)
    return float(score)


def psnr(img_a: Image.Image, img_b: Image.Image) -> float:
    """Peak Signal-to-Noise Ratio between two RGB PIL images (dB).

    Args:
        img_a: reference image.
        img_b: comparison image; must have the same spatial dimensions as img_a.

    Returns:
        PSNR in dB; returns float("inf") for identical images.

    Raises:
        ValueError: if the images differ in spatial dimensions.
    """
    a, b = _to_numpy_rgb(img_a), _to_numpy_rgb(img_b)
    _check_shape(a, b)
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def lpips_score(img_a: Image.Image, img_b: Image.Image) -> float:
    """Learned Perceptual Image Patch Similarity (AlexNet backbone).

    Args:
        img_a: reference image.
        img_b: comparison image; must have the same spatial dimensions as img_a.

    Returns:
        LPIPS distance ≥ 0; near 0 for identical images.

    Raises:
        ValueError: if the images differ in spatial dimensions.
    """
    a_np, b_np = _to_numpy_rgb(img_a), _to_numpy_rgb(img_b)
    _check_shape(a_np, b_np)

    def _to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
        # lpips expects (1, 3, H, W) float32 in [-1, 1]
        t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t

    a_t = _to_lpips_tensor(a_np)
    b_t = _to_lpips_tensor(b_np)

    model = _get_lpips_model()
    with torch.no_grad():
        dist = model(a_t, b_t)
    return float(dist.item())


def clip_score(image: Image.Image, prompt: str) -> float:
    """CLIP cosine similarity between an image and a text prompt.

    Measures how well the generated image matches the intended style/content
    described by *prompt*. Uses openai/clip-vit-base-patch32.

    Args:
        image:  RGB PIL image to evaluate.
        prompt: target text prompt (e.g. the gen_prompt used for generation).

    Returns:
        Cosine similarity ∈ [-1, 1]; higher = image better matches prompt.
    """
    model, processor = _get_clip()
    inputs = processor(
        text=[prompt],
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        img_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_embeds = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
    img_embeds = F.normalize(img_embeds, dim=-1)
    txt_embeds = F.normalize(txt_embeds, dim=-1)
    return float((img_embeds * txt_embeds).sum().item())


def dino_distance(img_a: Image.Image, img_b: Image.Image) -> float:
    """Structural distance between two images using DINO CLS-token features.

    Extracts the [CLS] token from facebook/dino-vits8 for each image and
    returns 1 − cosine_similarity. Lower = more structurally similar.
    Matches the evaluation protocol in PnP Figure 9.

    Args:
        img_a: first RGB PIL image (typically the source / original).
        img_b: second RGB PIL image (typically the generated output).

    Returns:
        Distance ∈ [0, 2]; 0 for structurally identical images.
    """
    model, extractor = _get_dino()
    inputs_a = extractor(images=img_a.convert("RGB"), return_tensors="pt")
    inputs_b = extractor(images=img_b.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        feat_a = model(**inputs_a).last_hidden_state[:, 0, :]  # CLS token
        feat_b = model(**inputs_b).last_hidden_state[:, 0, :]
    feat_a = F.normalize(feat_a, dim=-1)
    feat_b = F.normalize(feat_b, dim=-1)
    cos_sim = float((feat_a * feat_b).sum().item())
    return float(1.0 - cos_sim)
