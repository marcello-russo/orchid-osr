import { describe, it, expect } from "vitest";
import { f1Color, aurocColor, heatColor, heatTextColor } from "../metrics-colors";

describe("heatColor", () => {
  it("is red at 0", () => {
    expect(heatColor(0)).toContain("30.0");
  });

  it("is yellow at 0.5", () => {
    expect(heatColor(0.5)).toContain("100.0");
  });

  it("is green at 1", () => {
    expect(heatColor(1)).toContain("145.0");
  });

  it("clamps out-of-range values", () => {
    expect(heatColor(-1)).toBe(heatColor(0));
    expect(heatColor(2)).toBe(heatColor(1));
  });
});

describe("heatTextColor", () => {
  it("is dark ink on the yellow band", () => {
    expect(heatTextColor(0.5)).not.toBe("#ffffff");
  });

  it("is white on saturated red/green", () => {
    expect(heatTextColor(0)).toBe("#ffffff");
    expect(heatTextColor(1)).toBe("#ffffff");
  });
});

describe("f1Color", () => {
  it("returns valid oklch string", () => {
    expect(f1Color(0.5)).toMatch(/^oklch\(/);
  });

  it("maps 0/1 to red/green", () => {
    expect(f1Color(0)).toContain("30.0");
    expect(f1Color(1)).toContain("145.0");
  });
});

describe("aurocColor", () => {
  it("uses the same 0-1 red-yellow-green scale", () => {
    expect(aurocColor(0.5)).toContain("100.0");
    expect(aurocColor(0.8)).not.toBe(aurocColor(0.5));
  });
});
