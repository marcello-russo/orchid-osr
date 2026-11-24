import { describe, expect, it } from "vitest";
import { formatDate, formatNumber, formatPercent, speciesLabel } from "../format";

describe("formatNumber / formatPercent", () => {
  it("usa la virgola in italiano", () => {
    expect(formatPercent(0.833, 2, "it")).toBe("83,30%");
    expect(formatNumber(0.833, 3, "it")).toBe("0,833");
  });

  it("usa il punto in inglese", () => {
    expect(formatPercent(0.833, 2, "en")).toBe("83.30%");
    expect(formatNumber(0.833, 3, "en")).toBe("0.833");
  });

  it("non produce NaN in pagina", () => {
    expect(formatPercent(Number.NaN)).toBe("—");
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe("—");
  });

  it("formatta le date nella lingua richiesta", () => {
    expect(formatDate("2026-11-20T10:30:00", "it")).toContain("2026");
    expect(formatDate("non-una-data", "it")).toBe("—");
  });
});

describe("speciesLabel", () => {
  it("scrive il nome come nella tesi, senza underscore", () => {
    expect(speciesLabel("O. sphegodes_Palena")).toBe("O. sphegodes Palena");
  });
  it("lascia intatti i nomi già puliti", () => {
    expect(speciesLabel("O. garganica")).toBe("O. garganica");
  });
});
