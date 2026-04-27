# app/core/models/dinov2.py
import torch
import torch.nn as nn
from typing import Literal
from app.core.models.base import ClassifierModel, ModelConfig

_DINOV2_MODEL_IDS = {
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base": "facebook/dinov2-base",
}

_DINOV2_EMBEDDING_DIM = {
    "dinov2_small": 384,
    "dinov2_base": 768,
}


class DINOv2Config(ModelConfig):
    variant: Literal["dinov2_small", "dinov2_base"] = "dinov2_small"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    freeze_backbone: bool = True

    def create_model(self) -> "DINOv2Classifier":
        return DINOv2Classifier(self)


class DINOv2Classifier(ClassifierModel):

    def __init__(self, config: DINOv2Config):
        super().__init__(config)
        self._variant = config.variant
        self._embedding_dim = _DINOV2_EMBEDDING_DIM[config.variant]
        self._backbone = self._create_backbone(config)
        self._head = nn.Linear(self._embedding_dim, config.num_classes)

    def _create_backbone(self, config: DINOv2Config):
        from transformers import Dinov2Model
        from peft import LoraConfig, get_peft_model

        model_id = _DINOV2_MODEL_IDS[config.variant]
        backbone = Dinov2Model.from_pretrained(model_id)

        if config.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Apply LoRA to attention query/value projections
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
        outputs = self._backbone(x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self._head(cls_token)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self._backbone(x)
        return outputs.last_hidden_state[:, 0]

    def load_weights(self, path: str) -> None:
        import os
        adapter_dir = os.path.splitext(path)[0] + "_adapter"
        if os.path.isdir(adapter_dir):
            from peft import PeftModel
            base_model_id = _DINOV2_MODEL_IDS[self._variant]
            from transformers import Dinov2Model
            base = Dinov2Model.from_pretrained(base_model_id)
            self._backbone = PeftModel.from_pretrained(base, adapter_dir)
            self._head.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            state_dict = torch.load(path, map_location="cpu")
            self.load_state_dict(state_dict)

    def save_weights(self, path: str) -> None:
        import os
        adapter_dir = os.path.splitext(path)[0] + "_adapter"
        self._backbone.save_pretrained(adapter_dir)
        torch.save(self._head.state_dict(), path)

    @property
    def model_name(self) -> str:
        return self._variant

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
