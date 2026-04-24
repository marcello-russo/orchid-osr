// frontend/src/features/explorer/components/FoldSelector.tsx

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FOLD_COLORS } from "../types";

interface FoldSelectorProps {
  folds: { fold: number }[];
  selectedFold: number | null; // null = "all"
  onSelect: (fold: number | null) => void;
}

export function FoldSelector({ folds, selectedFold, onSelect }: FoldSelectorProps) {
  return (
    <Tabs
      value={selectedFold === null ? "all" : String(selectedFold)}
      onValueChange={(v) => onSelect(v === "all" ? null : Number(v))}
    >
      <div className="overflow-x-auto">
        <TabsList className="w-max">
          <TabsTrigger value="all">All Folds</TabsTrigger>
          {folds.map((f, i) => (
            <TabsTrigger key={f.fold} value={String(f.fold)}>
              <span
                className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full shrink-0"
                style={{ backgroundColor: FOLD_COLORS[i % FOLD_COLORS.length] }}
              />
              Fold {f.fold}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>
    </Tabs>
  );
}
