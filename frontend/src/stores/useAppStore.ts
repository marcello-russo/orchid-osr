// frontend/src/stores/useAppStore.ts

import { create } from "zustand";
import { persist } from "zustand/middleware";

type Theme = "light" | "dark";

function getSystemTheme(): Theme {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function getDefaultSidebarOpen(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(min-width: 768px)").matches;
}

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

interface AppState {
  theme: Theme;
  sidebarOpen: boolean;
  toggleTheme: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      theme: getSystemTheme(),
      sidebarOpen: getDefaultSidebarOpen(),

      toggleTheme: () => {
        const next = get().theme === "dark" ? "light" : "dark";
        applyTheme(next);
        set({ theme: next });
      },

      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      toggleSidebar: () => set({ sidebarOpen: !get().sidebarOpen }),
    }),
    {
      name: "orchid-app-store",
      partialize: (state) => ({ theme: state.theme }),
      onRehydrateStorage: () => (state) => {
        if (state) applyTheme(state.theme);
      },
    }
  )
);
