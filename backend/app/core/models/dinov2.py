# app/core/models/dinov2.py
import logging
from typing import Literal

import torch
from torch import nn

from app.core.models.base import ClassifierModel, ModelConfig

logger = logging.getLogger(__name__)

_DINOV2_MODEL_IDS = {
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base": "facebook/dinov2-base",
}

_DINOV2_EMBEDDING_DIM = {
    "dinov2_small": 384,
    "dinov2_base": 768,
}

# Multi-grano embedding = CLS final + CLS mid-layer + attention-pooled patches
_MULTI_GRAIN_DIM = {
    "dinov2_small": 384 + 384 + 384,  # 1152
    "dinov2_base": 768 + 768 + 768,  # 2304
}


class DINOv2Config(ModelConfig):
    variant: Literal["dinov2_small", "dinov2_base"] = "dinov2_small"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    freeze_backbone: bool = True
    multi_grain: bool = False
    full_finetune: bool = False

    def create_model(self) -> "DINOv2Classifier":
        return DINOv2Classifier(self)


class DINOv2Classifier(ClassifierModel):
    def __init__(self, config: DINOv2Config):
        super().__init__(config)
        self._variant = config.variant
        self._multi_grain = config.multi_grain
        if self._multi_grain:
            self._embedding_dim = _MULTI_GRAIN_DIM[config.variant]
        else:
            self._embedding_dim = _DINOV2_EMBEDDING_DIM[config.variant]
        self._backbone = self._create_backbone(config)
        self._head = nn.Linear(self._embedding_dim, config.num_classes)

    def _create_backbone(self, config: DINOv2Config):
        from transformers import Dinov2Model

        model_id = _DINOV2_MODEL_IDS[config.variant]
        backbone = Dinov2Model.from_pretrained(model_id)

        if config.full_finetune:
            # Full fine-tuning: no LoRA, backbone unfrozen
            logger.info("  DINOv2 full fine-tuning: backbone unfrozen, no LoRA")
            return backbone

        if config.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=config.lora_dropout,
        )
        backbone = get_peft_model(backbone, lora_config)
        return backbone

    @classmethod
    def from_config(cls, config: DINOv2Config) -> "DINOv2Classifier":
        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embeddings(x)
        if self._config.normalize_embeddings:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return self._head(emb)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if not self._multi_grain:
            outputs = self._backbone(x)
            return outputs.last_hidden_state[:, 0]

        outputs = self._backbone(x, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state

        # Global: CLS token from final layer
        cls_final = last_hidden[:, 0]

        # Local: attention-weighted patch pooling on final layer
        patches = last_hidden[:, 1:]
        # Simple mean pooling as local feature (attention maps not exposed in eval)
        patch_pooled = patches.mean(dim=1)

        # Regional: CLS token from mid-layer (layer 6 for both variants)
        mid_idx = min(6, len(outputs.hidden_states) - 1) if outputs.hidden_states else None
        if mid_idx is not None:
            cls_mid = outputs.hidden_states[mid_idx][:, 0]
        else:
            cls_mid = cls_final.new_zeros(cls_final.shape[0], cls_final.shape[1] // 2)

        return torch.cat([cls_final, patch_pooled, cls_mid], dim=1)

    def _get_backbone(self):
        from transformers import Dinov2Model

        return Dinov2Model.from_pretrained(_DINOV2_MODEL_IDS[self._variant])

    def _rebuild_head_for_dim(self, in_features: int) -> None:
        self._head = nn.Linear(in_features, self._config.num_classes)
        self._embedding_dim = in_features
        base = _DINOV2_EMBEDDING_DIM[self._variant]
        self._multi_grain = in_features > base

    def load_weights(self, path: str) -> None:
        import os

        adapter_dir = os.path.splitext(path)[0] + "_adapter"
        if os.path.isdir(adapter_dir):
            from peft import PeftModel

            base = self._get_backbone()
            self._backbone = PeftModel.from_pretrained(base, adapter_dir)
            head_sd = torch.load(path, map_location="cpu", weights_only=True)
            self._head.load_state_dict(head_sd)
            return

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        keys = set(state_dict.keys())

        # Detect head shape from checkpoint
        ckpt_head_key = next((k for k in keys if k.endswith("_head.weight") or k == "weight"), None)
        ckpt_head_w = state_dict.get(ckpt_head_key) if ckpt_head_key else None
        if ckpt_head_w is not None and ckpt_head_w.shape[1] != self._head.weight.shape[1]:
            logger.info(
                "DINOv2 checkpoint head dim %d != current %d — adapting",
                ckpt_head_w.shape[1],
                self._head.weight.shape[1],
            )
            self._rebuild_head_for_dim(ckpt_head_w.shape[1])

        # Case 1: full model state_dict (backbone + head)
        if any(k.startswith("_backbone.") for k in keys):
            has_peft = any("base_model.model" in k for k in keys)
            if not has_peft:
                backbone = self._get_backbone()
                self._backbone = backbone
            result = self.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                logger.warning("DINOv2 load_weights missing keys: %s", result.missing_keys)
            if result.unexpected_keys:
                logger.warning("DINOv2 load_weights unexpected keys: %s", result.unexpected_keys)
            return

        # Case 2: only head weights (weight + bias) — backbone from base model
        head_w = state_dict.get("weight")
        head_b = state_dict.get("bias")
        if head_w is not None and head_b is not None and len(keys) == 2:
            if head_w.shape != self._head.weight.shape or head_b.shape != self._head.bias.shape:
                self._rebuild_head_for_dim(head_w.shape[1])
            self._head.weight.data.copy_(head_w)
            self._head.bias.data.copy_(head_b)
            return

        # Case 3: head weights with _head. prefix
        head_sd = {k.replace("_head.", ""): v for k, v in state_dict.items() if k.startswith("_head.")}
        if head_sd:
            kw = head_sd.get("weight")
            if kw is not None and kw.shape[1] != self._head.weight.shape[1]:
                self._rebuild_head_for_dim(kw.shape[1])
            self._head.load_state_dict(head_sd)
            return

        result = self.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "DINOv2 load_weights fallback: %d missing, %d unexpected",
                len(result.missing_keys),
                len(result.unexpected_keys),
            )

    def save_weights(self, path: str) -> None:
        import os

        if self._config.full_finetune:
            # Full fine-tuning: save complete model state dict
            torch.save(self.state_dict(), path)
            logger.info(f"  Full FT checkpoint saved: {path}")
            return

        adapter_dir = os.path.splitext(path)[0] + "_adapter"
        self._backbone.save_pretrained(adapter_dir)
        torch.save(self._head.state_dict(), path)

    @property
    def model_name(self) -> str:
        return self._variant

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _unfreeze_head(self) -> None:
        for param in self._head.parameters():
            param.requires_grad = True
