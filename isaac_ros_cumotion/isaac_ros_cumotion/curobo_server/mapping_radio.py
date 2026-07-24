"""C-RADIO vision-language feature model for semantic mapping.

Ported verbatim from ``mapper_node.py`` — gated behind the
``enable_feature_mapping`` parameter.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


RADIO_MODEL_NAME = "c-radio_v3-B"


class CRadioInference:
    def __init__(self, device: str = "cuda:0", text_adaptor_name: Optional[str] = None):
        self.device = device
        adaptor_names = [text_adaptor_name] if text_adaptor_name else None
        hub_version = RADIO_MODEL_NAME.strip().lower()
        self.model = (
            torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                source="github",
                version=hub_version,
                progress=True,
                skip_validation=True,
                adaptor_names=adaptor_names,
            )
            .eval()
            .to(device)
        )
        self.patch_size = int(getattr(self.model, "patch_size", 16))

        self.text_adaptor = None
        self.tokenizer = None
        self._encode_text_fn = None
        if text_adaptor_name is not None:
            self.text_adaptor = self._resolve_text_adaptor(text_adaptor_name)
            self.tokenizer = getattr(self.text_adaptor, "tokenizer", None)
            self._encode_text_fn = getattr(self.text_adaptor, "encode_text", None)

    def _resolve_text_adaptor(self, adaptor_name: str):
        for attr in ("adaptors", "adapters", "_adaptors"):
            registry = getattr(self.model, attr, None)
            if registry is not None and adaptor_name in registry:
                return registry[adaptor_name]
        available = {}
        for attr in ("adaptors", "adapters", "_adaptors"):
            registry = getattr(self.model, attr, None)
            if registry is not None:
                available[attr] = list(registry.keys())
        raise RuntimeError(
            f"Could not find adaptor '{adaptor_name}' on RADIO model. "
            f"Available: {available or 'none'}"
        )

    def _project_through_text_adaptor(self, features: torch.Tensor) -> torch.Tensor:
        if self.text_adaptor is None:
            raise RuntimeError("No text adaptor loaded")
        for attr in ("head_mlp", "feat_mlp", "head"):
            sub = getattr(self.text_adaptor, attr, None)
            if sub is not None and callable(sub):
                return sub(features)
        if callable(self.text_adaptor):
            try:
                out = self.text_adaptor(features)
            except TypeError:
                summary = features.mean(dim=0, keepdim=True)
                out = self.text_adaptor(summary, features.unsqueeze(0))
                if isinstance(out, tuple):
                    out = out[1]
                    if out.dim() == 3:
                        out = out[0]
            if isinstance(out, tuple):
                out = out[1] if len(out) > 1 else out[0]
            return out
        raise RuntimeError(
            f"Adaptor {type(self.text_adaptor).__name__} has no known entry point"
        )

    @torch.inference_mode()
    def encode_text(self, text) -> torch.Tensor:
        if self.tokenizer is None or self._encode_text_fn is None:
            raise RuntimeError("No text adaptor loaded")
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text)
        if hasattr(tokens, "to"):
            tokens = tokens.to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                feats = self._encode_text_fn(tokens, normalize=True)
            except TypeError:
                feats = self._encode_text_fn(tokens)
                feats = F.normalize(feats, dim=-1)
        return feats

    @torch.inference_mode()
    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self._project_through_text_adaptor(features)
            out = F.normalize(out, dim=-1)
        return out

    @torch.inference_mode()
    def extract_patch_features(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        H, W = rgb_uint8.shape[:2]
        target_h, target_w = self.model.get_nearest_supported_resolution(H, W)
        img = rgb_uint8.permute(2, 0, 1).float() / 255.0
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        output = self.model(img)
        if isinstance(output, dict):
            output = output["backbone"]
        features = getattr(output, "features", None)
        if features is None:
            _, features = output
        ps = self.patch_size
        return features[0].view(target_h // ps, target_w // ps, -1).contiguous()
