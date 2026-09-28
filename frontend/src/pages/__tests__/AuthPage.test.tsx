import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { MemoryRouter } from "react-router-dom";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}));

vi.mock("@/lib/api", () => ({
  login: vi.fn(),
  register: vi.fn(),
}));

vi.mock("@/stores/useAppStore", () => ({
  useAppStore: () => ({ setAuth: vi.fn() }),
}));

import AuthPage from "../AuthPage";

function renderAt(entry: string) {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <AuthPage />
    </MemoryRouter>,
  );
}

describe("AuthPage ?invite= prefill", () => {
  it("precompila il codice e apre la registrazione", () => {
    renderAt("/auth?invite=inv-test-1");
    // il campo invito esiste solo in modalità registrazione
    expect(screen.getByDisplayValue("INV-TEST-1")).toBeInTheDocument();
  });

  it("senza parametro resta sul login", () => {
    renderAt("/auth");
    // niente campo invito = modalità login
    expect(screen.queryByPlaceholderText("auth.inviteCode")).not.toBeInTheDocument();
  });
});
