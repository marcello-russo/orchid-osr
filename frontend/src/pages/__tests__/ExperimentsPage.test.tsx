import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}));

vi.mock("@/features/explorer/components/BenchmarkMatrix", () => ({
  BenchmarkMatrix: () => React.createElement("div", { "data-testid": "benchmark-matrix" }, "Matrix"),
}));

vi.mock("@/lib/api", () => ({
  fetchExperiments: vi.fn(),
}));

import { fetchExperiments } from "@/lib/api";

const mockedFetch = vi.mocked(fetchExperiments);
import { ExperimentsPage } from "../ExperimentsPage";

const mockExperiments = [
  {
    id: "exp6_resnet18",
    name: "Exp 6 ResNet-18",
    model_variant: "resnet18",
    type: "kfold" as const,
    status: "completed" as const,
    num_folds: 5,
    folds_completed: 5,
    test_accuracy: 0.82,
    mean_f1: 0.85,
    std_f1: 0.02,
    timestamp: "2026-01-01",
    is_ncd: false,
    ncd_auroc: null,
    ncd_clustering_nmi: null,
  },
  {
    id: "exp7_resnet18",
    name: "Exp 7 ResNet-18",
    model_variant: "resnet18",
    type: "kfold" as const,
    status: "completed" as const,
    num_folds: 5,
    folds_completed: 5,
    test_accuracy: 0.88,
    mean_f1: 0.91,
    std_f1: 0.01,
    timestamp: "2026-02-01",
    is_ncd: false,
    ncd_auroc: null,
    ncd_clustering_nmi: null,
  },
];

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
}

describe("ExperimentsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedFetch.mockResolvedValue(mockExperiments);
  });

  it("shows loading skeletons initially", () => {
    mockedFetch.mockReturnValue(new Promise(() => {}));
    const { container } = render(<ExperimentsPage />, {
      wrapper: createWrapper(),
    });
    expect(
      container.querySelectorAll('[data-slot="skeleton"]').length,
    ).toBeGreaterThan(0);
  });

  it("shows experiments data when loaded", async () => {
    render(<ExperimentsPage />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByTestId("benchmark-matrix")).toBeInTheDocument();
    });
    expect(screen.getByText("experiments.title")).toBeInTheDocument();
  });

  it("shows error state with retry button", async () => {
    mockedFetch.mockRejectedValue(new Error("Network error"));
    render(<ExperimentsPage />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText("common.error")).toBeInTheDocument();
      expect(screen.getByText("common.retry")).toBeInTheDocument();
    });
  });

  it("toggles to matrix view", async () => {
    const user = userEvent.setup();
    render(<ExperimentsPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByTestId("benchmark-matrix")).toBeInTheDocument();
    });

    await user.click(screen.getByText("experiments.view_table"));
    expect(screen.queryByTestId("benchmark-matrix")).not.toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();

    await user.click(screen.getByText("experiments.view_matrix"));
    expect(screen.getByTestId("benchmark-matrix")).toBeInTheDocument();
  });

  it("selects experiments for comparison", async () => {
    const user = userEvent.setup();
    render(<ExperimentsPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByTestId("benchmark-matrix")).toBeInTheDocument();
    });
    await user.click(screen.getByText("experiments.view_table"));

    await waitFor(() => {
      expect(screen.getByRole("table")).toBeInTheDocument();
    });

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(2);

    await user.click(checkboxes[0]);
    expect(checkboxes[0]).toBeChecked();

    await user.click(checkboxes[1]);
    expect(checkboxes[1]).toBeChecked();
  });

  it("shows compare button when 2+ selected", async () => {
    const user = userEvent.setup();
    render(<ExperimentsPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByTestId("benchmark-matrix")).toBeInTheDocument();
    });
    await user.click(screen.getByText("experiments.view_table"));

    await waitFor(() => {
      expect(screen.getByRole("table")).toBeInTheDocument();
    });

    expect(screen.queryByText(/experiments\.compare/)).not.toBeInTheDocument();

    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[0]);
    await user.click(checkboxes[1]);

    expect(
      screen.getByRole("link", { name: /experiments\.compare/ }),
    ).toBeInTheDocument();
    expect(screen.getByText(/\(2\)/)).toBeInTheDocument();
  });
});
