// frontend/src/components/layout/Layout.tsx

import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { useAppStore } from "@/stores/useAppStore";
import { cn } from "@/lib/utils";

export function Layout() {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <Header />
      <main
        className={cn(
          "pt-14 transition-all duration-300",
          sidebarOpen ? "pl-56" : "pl-16"
        )}
      >
        <div className="mx-auto max-w-7xl p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
