// frontend/src/components/layout/Layout.tsx

import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { useAppStore } from "@/stores/useAppStore";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { cn } from "@/lib/utils";

export function Layout() {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen);
  const isDesktop = useMediaQuery("(min-width: 768px)");

  return (
    <div className="min-h-screen bg-background">
      {isDesktop ? (
        <Sidebar />
      ) : (
        <MobileOverlay />
      )}
      <Header isDesktop={isDesktop} />
      <main
        className={cn(
          "transition-all duration-300",
          isDesktop
            ? sidebarOpen
              ? "pt-14 pl-56"
              : "pt-14 pl-16"
            : "pt-14"
        )}
      >
        <div className="mx-auto max-w-7xl p-4 sm:p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

function MobileOverlay() {
  const { sidebarOpen, setSidebarOpen } = useAppStore();

  return (
    <>
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 transition-opacity duration-200"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <div
        className={cn(
          "fixed left-0 top-0 z-50 h-full w-72 bg-card shadow-xl transition-transform duration-200 ease-out",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <Sidebar />
      </div>
    </>
  );
}
