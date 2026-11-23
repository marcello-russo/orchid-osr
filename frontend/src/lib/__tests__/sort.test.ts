import { describe, it, expect } from "vitest";
import { compareNullSafe } from "../sort";

describe("compareNullSafe", () => {
  describe("numbers", () => {
    it("asc: a < b returns -1", () => { expect(compareNullSafe(1, 2, "asc")).toBe(-1); });
    it("asc: a > b returns 1", () => { expect(compareNullSafe(2, 1, "asc")).toBe(1); });
    it("asc: a === b returns 0", () => { expect(compareNullSafe(1, 1, "asc")).toBe(0); });
    it("desc: a < b returns 1", () => { expect(compareNullSafe(1, 2, "desc")).toBe(1); });
    it("desc: a > b returns -1", () => { expect(compareNullSafe(2, 1, "desc")).toBe(-1); });
    it("desc: a === b returns 0", () => { expect(compareNullSafe(1, 1, "desc")).toBe(0); });
  });

  describe("strings", () => {
    it("asc: a < b returns negative", () => { expect(compareNullSafe("a", "b", "asc")).toBeLessThan(0); });
    it("asc: a > b returns positive", () => { expect(compareNullSafe("b", "a", "asc")).toBeGreaterThan(0); });
    it("desc: a < b returns positive", () => { expect(compareNullSafe("a", "b", "desc")).toBeGreaterThan(0); });
  });

  describe("null and undefined", () => {
    it("null vs value: null last asc", () => { expect(compareNullSafe(null, 1, "asc")).toBe(1); });
    it("value vs null: value first asc", () => { expect(compareNullSafe(1, null, "asc")).toBe(-1); });
    it("null vs value: null last desc", () => { expect(compareNullSafe(null, 1, "desc")).toBe(1); });
    it("value vs null: value first desc", () => { expect(compareNullSafe(1, null, "desc")).toBe(-1); });
    it("null vs null returns 0", () => { expect(compareNullSafe(null, null, "asc")).toBe(0); });
    it("undefined treated as null", () => { expect(compareNullSafe(undefined, 1, "asc")).toBe(1); });
    it("both undefined returns 0", () => { expect(compareNullSafe(undefined, undefined, "asc")).toBe(0); });
  });

  describe("default direction", () => {
    it("defaults to asc", () => { expect(compareNullSafe(1, 2)).toBe(-1); });
  });
});
