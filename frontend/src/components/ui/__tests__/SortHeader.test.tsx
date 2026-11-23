import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SortHeader } from "../SortHeader";

describe("SortHeader", () => {
  it("renders label", () => {
    render(<SortHeader label="F1" sortKey="mean_f1" activeKey="id" dir="asc" onToggle={vi.fn()} />);
    expect(screen.getByText("F1")).toBeInTheDocument();
  });

  it("shows ↑ when active ascending", () => {
    render(<SortHeader label="F1" sortKey="mean_f1" activeKey="mean_f1" dir="asc" onToggle={vi.fn()} />);
    expect(screen.getByText("↑")).toBeInTheDocument();
  });

  it("shows ↓ when active descending", () => {
    render(<SortHeader label="F1" sortKey="mean_f1" activeKey="mean_f1" dir="desc" onToggle={vi.fn()} />);
    expect(screen.getByText("↓")).toBeInTheDocument();
  });

  it("does not show indicator when inactive", () => {
    render(<SortHeader label="F1" sortKey="mean_f1" activeKey="id" dir="asc" onToggle={vi.fn()} />);
    expect(screen.queryByText("↑")).not.toBeInTheDocument();
    expect(screen.queryByText("↓")).not.toBeInTheDocument();
  });

  it("calls onToggle on click", async () => {
    const onToggle = vi.fn();
    const user = userEvent.setup();
    render(<SortHeader label="F1" sortKey="mean_f1" activeKey="id" dir="asc" onToggle={onToggle} />);
    await user.click(screen.getByText("F1"));
    expect(onToggle).toHaveBeenCalledWith("mean_f1");
  });

  it("has aria-sort ascending when active asc", () => {
    const { container } = render(
      <SortHeader label="F1" sortKey="mean_f1" activeKey="mean_f1" dir="asc" onToggle={vi.fn()} />,
    );
    expect(container.querySelector("th")).toHaveAttribute("aria-sort", "ascending");
  });

  it("has aria-sort descending when active desc", () => {
    const { container } = render(
      <SortHeader label="F1" sortKey="mean_f1" activeKey="mean_f1" dir="desc" onToggle={vi.fn()} />,
    );
    expect(container.querySelector("th")).toHaveAttribute("aria-sort", "descending");
  });

  it("has no aria-sort when inactive", () => {
    const { container } = render(
      <SortHeader label="F1" sortKey="mean_f1" activeKey="id" dir="asc" onToggle={vi.fn()} />,
    );
    expect(container.querySelector("th")).not.toHaveAttribute("aria-sort");
  });
});
