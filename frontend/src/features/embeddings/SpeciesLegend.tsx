interface SpeciesLegendProps {
  species: string[];
  speciesColors: Record<string, string>;
  hidden: Set<string>;
  onToggle: (species: string) => void;
}

export function SpeciesLegend({ species, speciesColors, hidden, onToggle }: SpeciesLegendProps) {
  return (
    <div className="space-y-1.5">
      {species.map((s) => {
        const isHidden = hidden.has(s);
        return (
          <button
            key={s}
            onClick={() => onToggle(s)}
            className="flex items-center gap-2 w-full px-2.5 py-1.5 rounded-lg text-xs transition-colors hover:bg-muted/50 text-left"
          >
            <span
              className={`size-3 rounded-sm shrink-0 transition-opacity ${isHidden ? "opacity-30" : ""}`}
              style={{ backgroundColor: speciesColors[s] }}
            />
            <span className={`font-medium truncate ${isHidden ? "text-muted-foreground/50 line-through" : "text-foreground"}`}>
              {s}
            </span>
          </button>
        );
      })}
    </div>
  );
}
