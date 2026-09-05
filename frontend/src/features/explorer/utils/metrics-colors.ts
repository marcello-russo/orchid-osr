function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

type Oklch = [number, number, number];

// Anchors: bright red @0 → yellow @0.5 → vivid green @1.
const RED: Oklch = [0.55, 0.22, 30];
const YELLOW: Oklch = [0.85, 0.16, 100];
const GREEN: Oklch = [0.6, 0.18, 145];

function lerp(a: Oklch, b: Oklch, t: number): Oklch {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function fmt(c: Oklch): string {
  return `oklch(${c[0].toFixed(3)} ${c[1].toFixed(3)} ${c[2].toFixed(1)})`;
}

/** Red (0) → yellow (0.5) → green (1) background for both matrices. */
export function heatColor(value: number): string {
  const v = clamp01(value);
  return fmt(v < 0.5 ? lerp(RED, YELLOW, v * 2) : lerp(YELLOW, GREEN, (v - 0.5) * 2));
}

/** Readable foreground on heatColor: dark ink on the yellow band, white elsewhere. */
export function heatTextColor(value: number): string {
  const v = clamp01(value);
  return v > 0.32 && v < 0.68 ? "oklch(0.25 0.02 100)" : "#ffffff";
}

export function f1Color(f1: number): string {
  return heatColor(f1);
}

export function aurocColor(auroc: number): string {
  return heatColor(auroc);
}
