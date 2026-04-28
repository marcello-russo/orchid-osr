// frontend/src/features/explorer/components/ExperimentConfigCard.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import { TOOLTIPS } from "@/features/explorer/tooltipText";
import type { ExperimentConfig } from "@/lib/types";

interface ExperimentConfigCardProps {
  config: ExperimentConfig;
}

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
