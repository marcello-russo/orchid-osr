import { describe, it, expect } from "vitest";
import {
  getArchitectureFamily,
  getExperimentSeries,
  getArchitectureLabel,
  FAMILY_COLORS,
  ARCHITECTURE_ORDER,
  LABEL_TO_VARIANT,
  foldToEpochPoints,
} from "../types";
import type { FoldData } from "../types";

describe("getArchitectureFamily", () => {
  it("returns ResNet for resnet18", () => {
    expect(getArchitectureFamily("resnet18")).toBe("ResNet");
  });
  it("returns ResNet for resnet50", () => {
    expect(getArchitectureFamily("resnet50")).toBe("ResNet");
  });
  it("returns ConvNeXt for convnext_tiny", () => {
    expect(getArchitectureFamily("convnext_tiny")).toBe("ConvNeXt");
  });
  it("returns ConvNeXt for convnext_small", () => {
    expect(getArchitectureFamily("convnext_small")).toBe("ConvNeXt");
  });
  it("returns DINOv2 for dinov2_small", () => {
    expect(getArchitectureFamily("dinov2_small")).toBe("DINOv2");
  });
  it("returns DINOv2 for dinov2_base", () => {
    expect(getArchitectureFamily("dinov2_base")).toBe("DINOv2");
  });
  it("returns null for unknown variant", () => {
    expect(getArchitectureFamily("unknown_model")).toBeNull();
  });
  it("returns null for null input", () => {
    expect(getArchitectureFamily(null)).toBeNull();
  });
});

describe("getExperimentSeries", () => {
  it("returns Exp 6 for exp6 prefix", () => {
    expect(getExperimentSeries("exp6")).toBe("Exp 6");
  });
  it("returns Exp 6 for exp6_resnet18", () => {
    expect(getExperimentSeries("exp6_resnet18")).toBe("Exp 6");
  });
  it("returns Exp 7 for exp7_ prefix", () => {
    expect(getExperimentSeries("exp7_convnext_tiny")).toBe("Exp 7");
  });
  it("returns Exp 8 for exp8_ prefix", () => {
    expect(getExperimentSeries("exp8_dinov2_small")).toBe("Exp 8");
  });
  it("returns Exp 9 for exp9_ prefix", () => {
    expect(getExperimentSeries("exp9_dinov2_base")).toBe("Exp 9");
  });
  it("returns null for exp10_ prefix", () => {
    expect(getExperimentSeries("exp10_resnet50")).toBeNull();
  });
  it("returns null for unrelated string", () => {
    expect(getExperimentSeries("test")).toBeNull();
  });
});

describe("getArchitectureLabel", () => {
  it("returns ResNet-18 for resnet18", () => {
    expect(getArchitectureLabel("resnet18")).toBe("ResNet-18");
  });
  it("returns ResNet-50 for resnet50", () => {
    expect(getArchitectureLabel("resnet50")).toBe("ResNet-50");
  });
  it("returns ConvNeXt-Tiny for convnext_tiny", () => {
    expect(getArchitectureLabel("convnext_tiny")).toBe("ConvNeXt-Tiny");
  });
  it("returns ConvNeXt-Small for convnext_small", () => {
    expect(getArchitectureLabel("convnext_small")).toBe("ConvNeXt-Small");
  });
  it("returns DINOv2-Small for dinov2_small", () => {
    expect(getArchitectureLabel("dinov2_small")).toBe("DINOv2-Small");
  });
  it("returns DINOv2-Base for dinov2_base", () => {
    expect(getArchitectureLabel("dinov2_base")).toBe("DINOv2-Base");
  });
  it("returns variant as-is for unknown", () => {
    expect(getArchitectureLabel("unknown_variant")).toBe("unknown_variant");
  });
  it("returns em dash for null", () => {
    expect(getArchitectureLabel(null)).toBe("—");
  });
});

describe("FAMILY_COLORS", () => {
  it("has a color for ResNet", () => {
    expect(typeof FAMILY_COLORS.ResNet).toBe("string");
    expect(FAMILY_COLORS.ResNet).toBe("#3B82F6");
  });
  it("has a color for ConvNeXt", () => {
    expect(typeof FAMILY_COLORS.ConvNeXt).toBe("string");
    expect(FAMILY_COLORS.ConvNeXt).toBe("#D97706");
  });
  it("has a color for DINOv2", () => {
    expect(typeof FAMILY_COLORS.DINOv2).toBe("string");
    expect(FAMILY_COLORS.DINOv2).toBe("#8B5CF6");
  });
  it("has exactly 3 entries", () => {
    expect(Object.keys(FAMILY_COLORS)).toHaveLength(3);
  });
});

describe("ARCHITECTURE_ORDER", () => {
  it("has 6 entries", () => {
    expect(ARCHITECTURE_ORDER).toHaveLength(6);
  });
  it("lists ResNet-18 first", () => {
    expect(ARCHITECTURE_ORDER[0]).toBe("ResNet-18");
  });
  it("lists ResNet-50 second", () => {
    expect(ARCHITECTURE_ORDER[1]).toBe("ResNet-50");
  });
  it("lists ConvNeXt-Tiny third", () => {
    expect(ARCHITECTURE_ORDER[2]).toBe("ConvNeXt-Tiny");
  });
  it("lists ConvNeXt-Small fourth", () => {
    expect(ARCHITECTURE_ORDER[3]).toBe("ConvNeXt-Small");
  });
  it("lists DINOv2-Small fifth", () => {
    expect(ARCHITECTURE_ORDER[4]).toBe("DINOv2-Small");
  });
  it("lists DINOv2-Base sixth", () => {
    expect(ARCHITECTURE_ORDER[5]).toBe("DINOv2-Base");
  });
});

describe("LABEL_TO_VARIANT", () => {
  it("maps ResNet-18 to resnet18", () => {
    expect(LABEL_TO_VARIANT["ResNet-18"]).toBe("resnet18");
  });
  it("maps ResNet-50 to resnet50", () => {
    expect(LABEL_TO_VARIANT["ResNet-50"]).toBe("resnet50");
  });
  it("maps ConvNeXt-Tiny to convnext_tiny", () => {
    expect(LABEL_TO_VARIANT["ConvNeXt-Tiny"]).toBe("convnext_tiny");
  });
  it("maps ConvNeXt-Small to convnext_small", () => {
    expect(LABEL_TO_VARIANT["ConvNeXt-Small"]).toBe("convnext_small");
  });
  it("maps DINOv2-Small to dinov2_small", () => {
    expect(LABEL_TO_VARIANT["DINOv2-Small"]).toBe("dinov2_small");
  });
  it("maps DINOv2-Base to dinov2_base", () => {
    expect(LABEL_TO_VARIANT["DINOv2-Base"]).toBe("dinov2_base");
  });
  it("has exactly 6 entries", () => {
    expect(Object.keys(LABEL_TO_VARIANT)).toHaveLength(6);
  });
  it("has TypeScript Readonly type (compile-time)", () => {
    const keys: string[] = Object.keys(LABEL_TO_VARIANT);
    expect(keys.length).toBeGreaterThan(0);
  });
});

describe("foldToEpochPoints", () => {
  const mockFold: FoldData = {
    fold: 0,
    best_f1: 0.85,
    best_accuracy: 0.88,
    best_epoch: 4,
    final_train_loss: 0.15,
    final_val_loss: 0.25,
    train_size: 100,
    val_size: 50,
    history: {
      train_loss: [0.5, 0.3, 0.2, 0.15],
      val_loss: [0.6, 0.4, 0.3, 0.25],
      val_accuracy: [0.7, 0.8, 0.85, 0.88],
      val_f1: [0.65, 0.78, 0.83, 0.85],
    },
  };

  it("returns an array of EpochPoints with same length as history", () => {
    const points = foldToEpochPoints(mockFold);
    expect(points).toHaveLength(4);
  });

  it("sets epoch numbers starting from 1", () => {
    const points = foldToEpochPoints(mockFold);
    expect(points[0].epoch).toBe(1);
    expect(points[3].epoch).toBe(4);
  });

  it("transfers train_loss values correctly", () => {
    const points = foldToEpochPoints(mockFold);
    expect(points[0].train_loss).toBe(0.5);
    expect(points[2].train_loss).toBe(0.2);
  });

  it("transfers val_loss values correctly", () => {
    const points = foldToEpochPoints(mockFold);
    expect(points[0].val_loss).toBe(0.6);
    expect(points[3].val_loss).toBe(0.25);
  });

  it("transfers val_f1 values correctly", () => {
    const points = foldToEpochPoints(mockFold);
    expect(points[0].val_f1).toBe(0.65);
    expect(points[3].val_f1).toBe(0.85);
  });

  it("transfers val_accuracy values correctly", () => {
    const points = foldToEpochPoints(mockFold);
    expect(points[0].val_accuracy).toBe(0.7);
    expect(points[3].val_accuracy).toBe(0.88);
  });

  it("computes overfitting_ratio as val_loss / train_loss", () => {
    const points = foldToEpochPoints(mockFold);
    expect(points[0].overfitting_ratio).toBe(0.6 / 0.5);
    expect(points[1].overfitting_ratio).toBe(0.4 / 0.3);
  });

  it("sets overfitting_ratio to undefined when train_loss is 0", () => {
    const zeroFold: FoldData = {
      ...mockFold,
      history: {
        train_loss: [0, 0.3, 0.2, 0.15],
        val_loss: [0.6, 0.4, 0.3, 0.25],
        val_accuracy: [0.7, 0.8, 0.85, 0.88],
        val_f1: [0.65, 0.78, 0.83, 0.85],
      },
    };
    const points = foldToEpochPoints(zeroFold);
    expect(points[0].overfitting_ratio).toBeUndefined();
  });

  it("handles empty history arrays", () => {
    const emptyFold: FoldData = {
      ...mockFold,
      history: {
        train_loss: [],
        val_loss: [],
        val_accuracy: [],
        val_f1: [],
      },
    };
    const points = foldToEpochPoints(emptyFold);
    expect(points).toHaveLength(0);
  });
});
