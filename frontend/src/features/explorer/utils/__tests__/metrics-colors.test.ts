import { describe, it, expect } from "vitest";
import { f1Color, aurocColor } from "../metrics-colors";

describe("f1Color", () => {
  it("returns valid oklch string", () => {
    expect(f1Color(0.5)).toMatch(/^oklch\(/);
  });

  it("is green hue for high f1", () => {
    expect(f1Color(0.9)).toContain("145");
  });

  it("is amber hue for low f1", () => {
    expect(f1Color(0.3)).toContain("25");
  });

  it("lightness increases with f1", () => {
    const dark = f1Color(0);
    const bright = f1Color(1);
    expect(parseFloat(dark.split(" ")[0].slice(6))).toBeLessThan(
      parseFloat(bright.split(" ")[0].slice(6)),
    );
  });
});

describe("aurocColor", () => {
  it("returns gray for random (0.5)", () => {
    expect(aurocColor(0.5)).toBe("oklch(0.45 0.015 200)");
  });

  it("returns color for good (0.8)", () => {
    expect(aurocColor(0.8)).toMatch(/^oklch\(/);
    expect(aurocColor(0.8)).not.toBe(aurocColor(0.5));
  });

  it("returns gray for below random (0.3)", () => {
    expect(aurocColor(0.3)).toMatch(/^oklch\(0\.45/);
  });
});
