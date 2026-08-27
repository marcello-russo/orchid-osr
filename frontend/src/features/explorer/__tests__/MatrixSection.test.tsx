import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}));

const mockNavigate = vi.fn();
vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate,
}));

vi.mock("react-dom", () => ({
  createPortal: (content: React.ReactNode) => content,
}));

import { MatrixSection } from "../components/MatrixSection";
import type { ExperimentSummary } from "@/lib/types";

const mockExperiment: ExperimentSummary = {
  id: "exp1_ce_resnet18",
  name: "Exp 1 ResNet-18",
  model_variant: "resnet18",
  type: "kfold",
  status: "completed",
  num_folds: 5,
  folds_completed: 5,
  test_accuracy: 0.82,
  mean_f1: 0.85,
  std_f1: 0.02,
  timestamp: "2026-01-01",
  is_ncd: false,
  ncd_auroc: null,
  ncd_clustering_nmi: null,
};

const mockLookup = { resnet18: { "Exp 1": mockExperiment } };

const defaultProps = {
  lookup: mockLookup,
  architectureRows: ["ResNet-18"],
  columns: ["Exp 1"],
  accessor: (e: ExperimentSummary) => e.mean_f1,
  colorFn: (v: number) => `oklch(${0.15 + v * 0.6} 0.15 145)`,
  bestPerColumn: { "Exp 1": 0.85 },
  tooltipLabel: "F1 CV",
};

beforeEach(() => {
  mockNavigate.mockClear();
});

describe("MatrixSection", () => {
  it("renders table with model names", () => {
    render(<MatrixSection {...defaultProps} />);
    expect(screen.getByText("ResNet-18")).toBeInTheDocument();
  });

  it("renders formatted values with %", () => {
    render(<MatrixSection {...defaultProps} />);
    expect(screen.getByText("85.0%")).toBeInTheDocument();
  });

  it("renders best column star", () => {
    render(<MatrixSection {...defaultProps} />);
    expect(screen.getByText("★")).toBeInTheDocument();
  });

  it("renders empty state for null values", () => {
    render(
      <MatrixSection
        {...defaultProps}
        accessor={() => null}
        bestPerColumn={{}}
      />,
    );
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText("★")).not.toBeInTheDocument();
  });

  it("navigates on cell click", async () => {
    const user = userEvent.setup();
    render(<MatrixSection {...defaultProps} />);
    const cell = screen.getByRole("button");
    await user.click(cell);
    expect(mockNavigate).toHaveBeenCalledWith("/experiments/exp1_ce_resnet18");
  });

  it("renders title when provided", () => {
    render(<MatrixSection {...defaultProps} title="Test Title" />);
    expect(screen.getByText("Test Title")).toBeInTheDocument();
  });

  it("renders without title", () => {
    render(<MatrixSection {...defaultProps} />);
    expect(screen.queryByRole("heading")).not.toBeInTheDocument();
  });

  it("keyboard navigation on Enter", async () => {
    const user = userEvent.setup();
    render(<MatrixSection {...defaultProps} />);
    const cell = screen.getByRole("button");
    cell.focus();
    await user.keyboard("{Enter}");
    expect(mockNavigate).toHaveBeenCalledWith("/experiments/exp1_ce_resnet18");
  });

  it("keyboard navigation on Space", async () => {
    const user = userEvent.setup();
    render(<MatrixSection {...defaultProps} />);
    const cell = screen.getByRole("button");
    cell.focus();
    await user.keyboard(" ");
    expect(mockNavigate).toHaveBeenCalledWith("/experiments/exp1_ce_resnet18");
  });
});
