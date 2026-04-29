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
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3 md:grid-cols-4">
            {config.num_epochs && (
              <div>
                <span className="text-muted-foreground">Epochs</span>
                <InfoTooltip text={TOOLTIPS.epochs} />
                <span className="ml-1 font-mono text-foreground">{config.num_epochs}</span>
              </div>
            )}
            {config.batch_size && (
              <div>
                <span className="text-muted-foreground">Batch Size</span>
                <InfoTooltip text={TOOLTIPS.batchSize} />
                <span className="ml-1 font-mono text-foreground">{config.batch_size}</span>
              </div>
            )}
            {config.patience !== undefined && (
              <div>
                <span className="text-muted-foreground">Patience</span>
                <InfoTooltip text={TOOLTIPS.patience} />
                <span className="ml-1 font-mono text-foreground">{config.patience}</span>
              </div>
            )}
            <div>
              <span className="text-muted-foreground">LR</span>
              <InfoTooltip text={TOOLTIPS.learningRate} />
              <span className="ml-1 font-mono text-foreground">{config.effective_learning_rate || config.learning_rate}</span>
            </div>
            {config.optimizer && (
              <div>
                <span className="text-muted-foreground">Optimizer</span>
                <InfoTooltip text={TOOLTIPS.optimizer} />
                <span className="ml-1 font-mono text-foreground">{config.optimizer.toUpperCase()}</span>
              </div>
            )}
            {config.weight_decay !== undefined && (
              <div>
                <span className="text-muted-foreground">Weight Decay</span>
                <InfoTooltip text={TOOLTIPS.weightDecay} />
                <span className="ml-1 font-mono text-foreground">{config.weight_decay}</span>
              </div>
            )}
            {config.model_variant && (
              <div>
                <span className="text-muted-foreground">Source</span>
                <span className="ml-1 text-foreground">Registry ({config.model_variant})</span>
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3 md:grid-cols-4">
            {config.num_epochs && (
              <div>
                <span className="text-muted-foreground">Epochs</span>
                <InfoTooltip text={TOOLTIPS.epochs} />
                <span className="ml-1 font-mono text-foreground">{config.num_epochs}</span>
              </div>
            )}
            {config.batch_size && (
              <div>
                <span className="text-muted-foreground">Batch Size</span>
                <InfoTooltip text={TOOLTIPS.batchSize} />
                <span className="ml-1 font-mono text-foreground">{config.batch_size}</span>
              </div>
            )}
            {config.learning_rate && (
              <div>
                <span className="text-muted-foreground">LR</span>
                <InfoTooltip text={TOOLTIPS.learningRate} />
                <span className="ml-1 font-mono text-foreground">{config.learning_rate}</span>
              </div>
            )}
            {config.patience !== undefined && (
              <div>
                <span className="text-muted-foreground">Patience</span>
                <InfoTooltip text={TOOLTIPS.patience} />
                <span className="ml-1 font-mono text-foreground">{config.patience}</span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
