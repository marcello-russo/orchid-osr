import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSort } from "../useSort";

const items = [
  { id: "exp6", mean_f1: 0.85, test_accuracy: 0.82, model_variant: "resnet18" },
  { id: "exp7", mean_f1: 0.92, test_accuracy: 0.88, model_variant: "convnext" },
  { id: "exp9", mean_f1: 0.78, test_accuracy: 0.75, model_variant: "dinov2" },
];

describe("useSort", () => {
  it("sorts by default key ascending", () => {
    const { result } = renderHook(() => useSort(items, "id"));
    expect(result.current.sorted[0].id).toBe("exp6");
    expect(result.current.dir).toBe("asc");
  });

  it("sorts by default key descending", () => {
    const { result } = renderHook(() => useSort(items, "mean_f1", "desc"));
    expect(result.current.sorted[0].id).toBe("exp7"); // highest F1 first
  });

  it("toggles direction on same key", () => {
    const { result } = renderHook(() => useSort(items, "id"));
    act(() => result.current.toggle("id"));
    expect(result.current.dir).toBe("desc");
  });

  it("resets to asc on different key", () => {
    const { result } = renderHook(() => useSort(items, "id"));
    act(() => result.current.toggle("mean_f1"));
    expect(result.current.dir).toBe("asc");
    expect(result.current.key).toBe("mean_f1");
  });

  it("handles empty items", () => {
    const { result } = renderHook(() => useSort([], "id"));
    expect(result.current.sorted).toEqual([]);
  });

  it("handles single item", () => {
    const { result } = renderHook(() => useSort([items[0]], "id"));
    expect(result.current.sorted).toHaveLength(1);
  });

  it("sorts by model_variant", () => {
    const { result } = renderHook(() => useSort(items, "model_variant"));
    expect(result.current.sorted[0].model_variant).toBe("convnext");
  });
});
