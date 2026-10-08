"""Blue-Eye inference: image file(s) -> class + probabilities (safe / suggestive / explicit).

Command line
    python inference.py photo.jpg folder_of_images/ --device cuda --precision bf16
    python inference.py *.png --json > results.json

Python
    from inference import classify
    for r in classify(["photo.jpg"]):
        print(r["label"], r["probabilities"])

`--precision bf16` runs only the backbone under bfloat16 autocast; the head and the softmax stay
float32 in every mode. The default is float32 everywhere; bf16 is faster on GPUs with bfloat16 tensor
cores (NVIDIA Ampere or newer) and on recent CPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from modeling import BlueEyeClassifier, normalize, preprocess_uint8

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_CACHE: Dict[tuple, BlueEyeClassifier] = {}


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision in ("fp32", "float32"):
        return None
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError("precision must be fp32 or bf16, not %r" % precision)


def expand_paths(inputs: Iterable[Union[str, os.PathLike]]) -> List[str]:
    """Files are kept as given; folders are expanded to the image files they contain (sorted, recursive)."""
    out = []
    for p in inputs:
        p = os.fspath(p)
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                out += [os.path.join(root, f) for f in sorted(files)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        else:
            out.append(p)
    return out


def get_model(model: Union[str, BlueEyeClassifier, None] = None, device: Optional[str] = None) -> BlueEyeClassifier:
    """A loaded model: pass one through, or load (and cache) from a folder / Hub repo id (default: this folder)."""
    if isinstance(model, BlueEyeClassifier):
        return model
    device = device or _default_device()
    key = (model or _HERE, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = BlueEyeClassifier.from_pretrained(model or _HERE, device=device)
    return _MODEL_CACHE[key]


@torch.no_grad()
def classify(images: Sequence[Union[str, os.PathLike]], model: Union[str, BlueEyeClassifier, None] = None,
             device: Optional[str] = None, precision: str = "fp32", batch_size: int = 8) -> List[dict]:
    """Classify image files. Returns one dict per image, in order:
        {"path": ..., "label": "safe" | "suggestive" | "explicit",
         "probabilities": {"safe": p, "suggestive": p, "explicit": p}}
    or {"path": ..., "error": "..."} for a file that cannot be read."""
    device = device or _default_device()
    m = get_model(model, device)
    dev = next(m.parameters()).device
    ac = _autocast_dtype(precision)
    labels = m.labels
    results: List[Optional[dict]] = [None] * len(images)
    batch, idx = [], []

    def flush():
        if not batch:
            return
        x = normalize(torch.from_numpy(np.stack(batch)).to(dev), m.config)
        probs = m.predict_proba(x, autocast_dtype=ac).cpu().numpy()
        for j, pr in zip(idx, probs):
            results[j] = {"path": os.fspath(images[j]), "label": labels[int(pr.argmax())],
                          "probabilities": {c: float(pr[k]) for k, c in enumerate(labels)}}
        batch.clear(); idx.clear()

    for i, p in enumerate(images):
        try:
            batch.append(preprocess_uint8(p, m.config))
            idx.append(i)
        except Exception as e:  # unreadable or truncated file
            results[i] = {"path": os.fspath(p), "error": "%s: %s" % (type(e).__name__, e)}
        if len(batch) >= batch_size:
            flush()
    flush()
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Blue-Eye: classify images as safe / suggestive / explicit")
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--model", default=None, help="model folder or Hub repo id (default: this folder)")
    ap.add_argument("--device", default=None, help="cpu, cuda, cuda:1, mps (default: best available)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"],
                    help="backbone precision; the head is always float32 (default: fp32)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="print one JSON list instead of a table")
    a = ap.parse_args(argv)

    paths = expand_paths(a.inputs)
    if not paths:
        print("no images found", file=sys.stderr)
        return 1
    res = classify(paths, model=a.model, device=a.device, precision=a.precision, batch_size=a.batch_size)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for r in res:
            if "error" in r:
                print("%-10s %s  (%s)" % ("ERROR", r["path"], r["error"]))
            else:
                pr = r["probabilities"]
                print("%-10s safe %.3f  suggestive %.3f  explicit %.3f  %s"
                      % (r["label"], pr["safe"], pr["suggestive"], pr["explicit"], r["path"]))
    return 0 if all("error" not in r for r in res) else 2


if __name__ == "__main__":
    sys.exit(main())
