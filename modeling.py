"""Blue-Eye: a 3-class content-safety image classifier (safe / suggestive / explicit).

Architecture
    backbone  DINOv3 ViT-L/16 (Hugging Face `transformers` implementation, 24 layers, width 1024,
              4 register tokens, RoPE positions), run at 512 x 512 pixels -> 1 + 4 + 1024 tokens.
    pooling   concat(final CLS token, mean of every other final token)          -> 2048 features
    head      LayerNorm(2048) -> Linear(2048, 3)                                 -> logits

Precision
    The weights are float32. The backbone may run under bfloat16 autocast; the pooled features,
    the head and the softmax always run in float32, even if the module has been cast to a lower
    precision. Logits are returned in float32.

Only `torch`, `transformers` (>= 4.56, for DINOv3), `safetensors`, `Pillow` and `numpy` are needed.
`huggingface_hub` is needed only to load the model by repository id instead of a local folder.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"

ImageInput = Union[str, os.PathLike, Image.Image]


# ----------------------------------------------------------------------------------- preprocessing
def _resize_shortest_edge(img: Image.Image, size: int) -> Image.Image:
    """Bicubic resize so the shorter edge equals `size` (long edge truncated toward zero).

    Identical, pixel for pixel, to torchvision `Resize(size, InterpolationMode.BICUBIC)` on a PIL
    image, which is what the model was trained and evaluated with.
    """
    w, h = img.size
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = size, int(size * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.BICUBIC)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    """Centre crop to size x size; identical to torchvision `CenterCrop(size)` when the image is
    at least that large (always true after `_resize_shortest_edge` with a larger size)."""
    w, h = img.size
    if w < size or h < size:
        raise ValueError("image %dx%d is smaller than the %d crop" % (w, h, size))
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def load_image(image: ImageInput) -> Image.Image:
    """Open an image (path or PIL image) as RGB. EXIF orientation is not applied, matching training."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as im:
        return im.convert("RGB")


def preprocess_uint8(image: ImageInput, config: dict) -> np.ndarray:
    """One image -> uint8 array of shape (crop, crop, 3): RGB, shortest-edge bicubic resize, centre crop."""
    p = config["preprocessing"]
    img = load_image(image)
    img = _center_crop(_resize_shortest_edge(img, p["resize_shortest_edge"]), p["crop_size"])
    return np.array(img, dtype=np.uint8)


def normalize(batch_uint8: torch.Tensor, config: dict) -> torch.Tensor:
    """uint8 (N, H, W, 3) or (N, 3, H, W) -> float32 (N, 3, H, W), normalised with the ImageNet mean/std."""
    x = batch_uint8
    if x.dim() != 4:
        raise ValueError("expected a 4-d batch, got shape %s" % (tuple(x.shape),))
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)
    p = config["preprocessing"]
    mean = torch.tensor(p["image_mean"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(p["image_std"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1) * 255.0
    return (x.to(torch.float32) - mean) / std


def preprocess(images: Sequence[ImageInput], config: dict) -> torch.Tensor:
    """Images (paths or PIL images) -> float32 pixel tensor (N, 3, crop, crop) ready for the model."""
    arr = np.stack([preprocess_uint8(im, config) for im in images])
    return normalize(torch.from_numpy(arr), config)


# ----------------------------------------------------------------------------------- model
class BlueEyeHead(nn.Module):
    def __init__(self, in_features: int, num_labels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, eps=eps)
        self.classifier = nn.Linear(in_features, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Always float32, whatever the module's own dtype: the logits must not be rounded.
        f = features.to(torch.float32)
        f = F.layer_norm(f, self.norm.normalized_shape, self.norm.weight.to(torch.float32),
                         self.norm.bias.to(torch.float32), self.norm.eps)
        return F.linear(f, self.classifier.weight.to(torch.float32), self.classifier.bias.to(torch.float32))


def _build_backbone(config: dict, attn_implementation: Optional[str]) -> nn.Module:
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as e:  # pragma: no cover
        raise ImportError("Blue-Eye needs `transformers>=4.56` for the DINOv3 backbone") from e
    bcfg = dict(config["backbone_config"])
    model_type = bcfg.pop("model_type")
    try:
        hf_cfg = AutoConfig.for_model(model_type, **bcfg)
    except (KeyError, ValueError) as e:
        raise ImportError("this transformers version does not know %r; install transformers>=4.56" % model_type) from e
    impl = attn_implementation or config.get("attn_implementation", "sdpa")
    try:
        return AutoModel.from_config(hf_cfg, attn_implementation=impl)
    except (ValueError, ImportError):
        return AutoModel.from_config(hf_cfg, attn_implementation="eager")


class BlueEyeClassifier(nn.Module):
    """DINOv3 ViT-L/16 backbone + LayerNorm/Linear head. `forward` returns float32 logits (N, 3)."""

    def __init__(self, config: dict, attn_implementation: Optional[str] = None):
        super().__init__()
        self.config = config
        self.backbone = _build_backbone(config, attn_implementation)
        h = config["head"]
        self.head = BlueEyeHead(h["in_features"], config["num_labels"], h.get("layer_norm_eps", 1e-5))
        self.id2label = {int(k): v for k, v in config["id2label"].items()}

    @property
    def labels(self) -> List[str]:
        return [self.id2label[i] for i in range(len(self.id2label))]

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path_or_repo_id: str = ".", device: Union[str, torch.device, None] = None,
                        revision: Optional[str] = None, attn_implementation: Optional[str] = None,
                        token: Union[str, bool, None] = None) -> "BlueEyeClassifier":
        """Load from a local folder holding config.json + model.safetensors, or from a Hugging Face
        Hub repository id (requires `huggingface_hub`). Returns the model in eval mode."""
        folder = str(path_or_repo_id)
        if not os.path.isdir(folder):
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                raise FileNotFoundError("%r is not a folder, and huggingface_hub is not installed to "
                                        "download it" % folder) from e
            folder = snapshot_download(folder, revision=revision, token=token,
                                       allow_patterns=[CONFIG_NAME, WEIGHTS_NAME])
        with open(os.path.join(folder, CONFIG_NAME), encoding="utf-8") as f:
            config = json.load(f)
        model = cls(config, attn_implementation=attn_implementation)
        from safetensors.torch import load_file
        model.load_release_state_dict(load_file(os.path.join(folder, WEIGHTS_NAME)))
        model.eval()
        if device is not None:
            model.to(device)
        return model

    def load_release_state_dict(self, sd: dict) -> None:
        """Strictly load the release tensors (`backbone.layer.N.*` naming) into whichever layout the
        installed transformers version uses (`backbone.layer.N.*` in 4.x, `backbone.model.layer.N.*`
        in 5.x). Raises if any tensor is missing, unexpected or mis-shaped."""
        want = self.state_dict()
        nested = any(k.startswith("backbone.model.layer.") for k in want)
        out = {}
        for k, v in sd.items():
            if nested and k.startswith("backbone.layer."):
                k = "backbone.model." + k[len("backbone."):]
            out[k] = v
        missing = sorted(set(want) - set(out))
        unexpected = sorted(set(out) - set(want))
        if missing or unexpected:
            raise RuntimeError("weights do not match the architecture: %d missing (e.g. %s), %d unexpected (e.g. %s)"
                               % (len(missing), missing[:3], len(unexpected), unexpected[:3]))
        self.load_state_dict(out, strict=True)
        self.float()

    # ------------------------------------------------------------------ inference
    def forward(self, pixel_values: torch.Tensor, autocast_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """pixel_values: float (N, 3, 512, 512) from `preprocess`/`normalize`.
        autocast_dtype: None for float32, or torch.bfloat16 to run the BACKBONE under autocast.
        Returns float32 logits (N, 3); the pooling and the head always run in float32."""
        dev = pixel_values.device.type
        with torch.autocast(device_type=dev, dtype=autocast_dtype or torch.bfloat16,
                            enabled=autocast_dtype is not None):
            h = self.backbone(pixel_values=pixel_values).last_hidden_state
        with torch.autocast(device_type=dev, enabled=False):
            h = h.to(torch.float32)
            feats = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1)
            return self.head(feats)

    @torch.no_grad()
    def predict_proba(self, pixel_values: torch.Tensor, autocast_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """float32 class probabilities (N, 3), softmax in float32."""
        return torch.softmax(self.forward(pixel_values, autocast_dtype).to(torch.float32), dim=-1)


def load_model(path_or_repo_id: str = ".", device: Union[str, torch.device, None] = None, **kw) -> BlueEyeClassifier:
    return BlueEyeClassifier.from_pretrained(path_or_repo_id, device=device, **kw)
