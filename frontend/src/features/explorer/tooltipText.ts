// frontend/src/features/explorer/tooltipText.ts

export const TOOLTIPS = {
  optimizer:
    "Adam for ResNet (proven fine-tuning baseline). AdamW for ConvNeXt and DINOv2 — decoupled weight decay prevents overfitting on small datasets.",
  learningRate:
    "Fine-tuning LR from the model registry. ResNet: 0.001 (standard), ConvNeXt: 0.0001 (LayerNorm is sensitive to large gradients), DINOv2: 0.0002 (LoRA adapter).",
  weightDecay:
    "L2 regularization strength. 0 for ResNet (BatchNorm provides implicit regularization), 0.05 for ConvNeXt (ConvNeXt paper recipe), 0.01 for DINOv2 (light regularization on LoRA adapters).",
  patience:
    "Stops training after 15 consecutive epochs without val_loss improvement. Catches convergence without premature stopping.",
  batchSize:
    "Physical batch=8 (fits 4GB VRAM). Effective batch=32 via gradient accumulation (4 steps). Mathematically identical to batch=32 for cross-entropy loss.",
  epochs:
    "Maximum training duration. Early stopping terminates earlier via patience.",
  testAccuracy:
    "Overall fraction of correctly classified test images (300 total, 50 per class).",
  f1macro:
    "Harmonic mean of per-class F1 scores. Preferred over accuracy for imbalanced datasets.",
} as const;
