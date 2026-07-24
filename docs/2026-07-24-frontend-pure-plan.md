# Frontend Pure Functions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development

**Goal:** Testare tutte le funzioni pure in types.ts + colLabel/colAlign in ExperimentsPage.tsx

**Files:**
- Create: `frontend/src/features/explorer/__tests__/types.test.ts` (~15 test)
- Create: `frontend/src/pages/__tests__/ExperimentsPage.helpers.test.ts` (~5 test)

### Task 1: types.ts helpers

Testa: `getArchitectureFamily`, `getExperimentSeries`, `getArchitectureLabel`, `FAMILY_COLORS`, `ARCHITECTURE_ORDER`, `LABEL_TO_VARIANT`

```typescript
import { describe, it, expect } from "vitest";
import { getArchitectureFamily, getExperimentSeries, getArchitectureLabel, LABEL_TO_VARIANT } from "../types";

describe("getArchitectureFamily", () => {
  it("returns ResNet for resnet18", () => expect(getArchitectureFamily("resnet18")).toBe("ResNet"));
  it("returns ConvNeXt for convnext_tiny", () => expect(getArchitectureFamily("convnext_tiny")).toBe("ConvNeXt"));
  it("returns DINOv2 for dinov2_small", () => expect(getArchitectureFamily("dinov2_small")).toBe("DINOv2"));
  it("returns null for unknown", () => expect(getArchitectureFamily("unknown")).toBeNull());
  it("returns null for null", () => expect(getArchitectureFamily(null)).toBeNull());
});

describe("getExperimentSeries", () => {
  it("parses exp6", () => expect(getExperimentSeries("exp6_resnet18")).toBe("Exp 6"));
  it("parses exp10", () => expect(getExperimentSeries("exp10_dinov2")).toBe("Exp 10"));
  it("returns null for unknown", () => expect(getExperimentSeries("exp99_test")).toBeNull());
});

describe("getArchitectureLabel", () => {
  it("labels resnet18", () => expect(getArchitectureLabel("resnet18")).toBe("ResNet-18"));
  it("labels dinov2_base", () => expect(getArchitectureLabel("dinov2_base")).toBe("DINOv2-Base"));
  it("falls back to input for unknown", () => expect(getArchitectureLabel("custom_model")).toBe("custom_model"));
  it("returns em dash for null", () => expect(getArchitectureLabel(null)).toBe("—"));
});

describe("LABEL_TO_VARIANT", () => {
  it("maps ResNet-18 to resnet18", () => expect(LABEL_TO_VARIANT["ResNet-18"]).toBe("resnet18"));
  it("is readonly", () => expect(() => { (LABEL_TO_VARIANT as any)["X"] = "Y"; }).toThrow());
});
```

### Task 2: colLabel/colAlign helpers

I test per colLabel e colAlign DEVONO usare import dinamico. Per testarli senza montare ExperimentsPage, estrarre le funzioni in un file separato o testarle tramite renderHook.

Approccio: estrarle in `src/pages/experimentsHelpers.ts` e importarle nel test.

```typescript
import { describe, it, expect } from "vitest";
import { colLabel, colAlign } from "../experimentsHelpers";

const SORTABLE_COLUMNS = [
  { key: "id", labelKey: "experiments.table.experiment" },
  { key: "mean_f1", labelKey: "experiments.table.f1", align: "right" },
] as const;

describe("colLabel", () => {
  it("uses i18n key", () => {
    const t = (k: string) => `TRANS:${k}`;
    expect(colLabel(SORTABLE_COLUMNS[0], t)).toBe("TRANS:experiments.table.experiment");
  });
});

describe("colAlign", () => {
  it("returns align when present", () => expect(colAlign(SORTABLE_COLUMNS[1])).toBe("right"));
  it("returns undefined when not present", () => expect(colAlign(SORTABLE_COLUMNS[0])).toBeUndefined());
});
```

### Esecuzione

```bash
cd frontend && npx vitest run src/features/explorer/__tests__/types.test.ts src/pages/__tests__/ExperimentsPage.helpers.test.ts -v
```
Expected: ~20 test passano
