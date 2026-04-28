// frontend/src/features/explorer/components/ExperimentConfigCard.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import type { ExperimentConfig } from "@/lib/types";

interface ExperimentConfigCardProps {
  config: ExperimentConfig;
}

const TOOLTIPS = {
  optimizer:
    "Adam for ResNet (proven fine-tuning baseline). AdamW for ConvNeXt and DINOv2 — decoupled weight decay prevents overfitting on small datasets.",
  learningRate:
    "Fine-tuning LR from the model registry. ResNet: 0.001 (standard), ConvNeXt: 0.0001 (LayerNorm is sensitive to large gradients), DINOv2: 0.0002 (LoRA adapter).",
  weightDecay:
    "L2 regularization strength. 0 for ResNet (BatchNorm provides implicit regularization), 0.05 for ConvNeXt (ConvNeXt paper recipe), 0.01 for DINOv2 (light regularization on LoRA adapters).",
};

export function ExperimentConfigCard({ config }: ExperimentConfigCardProps) {
  const hasRecipe =
    config.optimizer !== undefined ||
    config.effective_learning_rate !== undefined ||
    config.weight_decay !== undefined;

  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-base">Training Recipe</CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        {hasRecipe ? (
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
            <div>
              <span className="text-muted-foreground">Optimizer</span>
              <InfoTooltip text={TOOLTIPS.optimizer} />
              <span className="ml-1 font-mono">{config.optimizer?.toUpperCase()}</span>
            </div>
            <div>
              <span className="text-muted-foreground">LR</span>
              <InfoTooltip text={TOOLTIPS.learningRate} />
              <span className="ml-1 font-mono">{config.effective_learning_rate}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Weight Decay</span>
              <InfoTooltip text={TOOLTIPS.weightDecay} />
              <span className="ml-1 font-mono">{config.weight_decay}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Source</span>
              <span className="ml-1">Model registry ({config.model_variant})</span>
            </div>
          </div>
        ) : (
          <div className="text-sm text-muted-foreground">
            <span className="font-mono">Adam, LR={config.learning_rate ?? 0.001}</span>
            <span className="ml-2 italic">(default recipe)</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
