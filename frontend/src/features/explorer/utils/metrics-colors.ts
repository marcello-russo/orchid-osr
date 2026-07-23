export function f1Color(f1: number): string {
  const l = 0.15 + f1 * 0.6;
  return `oklch(${l} 0.15 ${f1 > 0.6 ? 145 : 25})`;
}

export function aurocColor(auroc: number): string {
  const aboveRandom = Math.max(0, (auroc - 0.5) * 2);
  if (aboveRandom < 0.05) return "oklch(0.45 0.015 200)";
  const l = 0.35 + aboveRandom * 0.4;
  const chroma = 0.04 + aboveRandom * 0.14;
  return `oklch(${l} ${chroma} 145)`;
}
