// frontend/src/features/explorer/components/FoldSelector.tsx

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useTranslation } from "react-i18next";
import { useChartColors } from "../hooks/useChartColors";
import type { FoldData } from "@/lib/types";

interface FoldSelectorProps {
  folds: FoldData[];
  selectedFold: number | null; // null = "all"
  onSelect: (fold: number | null) => void;
}

export function FoldSelector({ folds, selectedFold, onSelect }: FoldSelectorProps) {
  const { t } = useTranslation();
  const chartColors = useChartColors();

  return (
    <Tabs
      value={selectedFold === null ? "all" : String(selectedFold)}
      onValueChange={(v) => onSelect(v === "all" ? null : Number(v))}
    >
      <div className="overflow-x-auto overflow-y-hidden">
        <TabsList className="w-max">
          <TabsTrigger value="all">{t("experiment_detail.fold_selector.all")}</TabsTrigger>
          {folds.map((f, i) => (
            <TabsTrigger key={f.fold} value={String(f.fold)}>
              <span
                className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full shrink-0"
                style={{ backgroundColor: chartColors[i % chartColors.length] }}
              />
              {t("experiment_detail.fold_selector.fold")}{f.fold}

            </TabsTrigger>
          ))}
        </TabsList>
      </div>
    </Tabs>
  );
}
