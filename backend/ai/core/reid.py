"""
Re-ID model wrapper (OSNet) for CPU inference.
"""
from __future__ import annotations

import numpy as np
import cv2

try:
    import torch
    import torchreid
except Exception:  # pragma: no cover - optional dependency
    torch = None
    torchreid = None


class ReIDModel:
    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cpu"):
        if torch is None or torchreid is None:
            raise ImportError("torchreid is not installed")

        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, img_bgr: np.ndarray) -> "torch.Tensor":
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Re-ID standard input size: 256x128 (HxW)
        img_rgb = cv2.resize(img_rgb, (128, 256))
        img = img_rgb.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # CHW
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def extract(self, img_bgr: np.ndarray) -> np.ndarray | None:
        if img_bgr is None or img_bgr.size == 0:
            return None
        x = self._preprocess(img_bgr)
        feat = self.model(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        feat = feat.detach().cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
