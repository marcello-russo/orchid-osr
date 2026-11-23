import type { TFunction } from "i18next";

export function getTooltips(t: TFunction) {
  return {
    optimizer: t("experiment_detail.tooltips.optimizer"),
    learningRate: t("experiment_detail.tooltips.learning_rate"),
    weightDecay: t("experiment_detail.tooltips.weight_decay"),
    patience: t("experiment_detail.tooltips.patience"),
    batchSize: t("experiment_detail.tooltips.batch_size"),
    epochs: t("experiment_detail.tooltips.epochs"),
    testAccuracy: t("experiment_detail.tooltips.test_accuracy"),
    f1macro: t("experiment_detail.tooltips.f1_macro"),
  };
}
