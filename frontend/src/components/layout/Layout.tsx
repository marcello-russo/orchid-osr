// frontend/src/components/layout/Layout.tsx

import { useEffect } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { useAppStore } from "@/stores/useAppStore";
import { cn } from "@/lib/utils";

export function Layout() {
  const { sidebarOpen, setSidebarOpen } = useAppStore();

  // Lock body scroll when drawer is open
  useEffect(() => {
    if (sidebarOpen) {
      document.body.style.overflow = "hidden";
      return () => { document.body.style.overflow = ""; };
    }
  }, [sidebarOpen]);

  // Close on Escape key
  useEffect(() => {
    if (!sidebarOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSidebarOpen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [sidebarOpen, setSidebarOpen]);

  return (
    <div className="min-h-screen bg-background">
      <Header />

      {/* Backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 transition-opacity duration-200"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar drawer */}
      <div
        id="sidebar"
        role="dialog"
        aria-modal="true"
        aria-label="Navigation"
        className={cn(
          "fixed left-0 top-0 z-50 h-full w-72 shadow-xl transition-transform duration-200 ease-out",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <Sidebar />
      </div>

      <main className="pt-14">
        <div className="mx-auto max-w-7xl p-4 sm:p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
