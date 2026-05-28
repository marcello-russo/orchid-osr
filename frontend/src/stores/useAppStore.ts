// frontend/src/stores/useAppStore.ts

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AuthUser } from "../lib/types";

type Theme = "light" | "dark";

function getSystemTheme(): Theme {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

interface AppState {
  theme: Theme;
  sidebarOpen: boolean;
  token: string | null;
  user: AuthUser | null;
  toggleTheme: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  setAuth: (token: string, user: AuthUser) => void;
  logout: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      theme: getSystemTheme(),
      sidebarOpen: false,
      token: null,
      user: null,

      toggleTheme: () => {
        const next = get().theme === "dark" ? "light" : "dark";
        applyTheme(next);
        set({ theme: next });
      },

      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      toggleSidebar: () => set({ sidebarOpen: !get().sidebarOpen }),

      setAuth: (token: string, user: AuthUser) => set({ token, user }),

      logout: () => set({ token: null, user: null }),
    }),
    {
      name: "orchid-app-store",
      partialize: (state) => ({
        theme: state.theme,
        token: state.token,
        user: state.user,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) applyTheme(state.theme);
      },
    }
  )
);
