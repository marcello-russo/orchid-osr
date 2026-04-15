// frontend/src/components/layout/Sidebar.tsx

import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  FlaskConical,
  ScanSearch,
  Box,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useAppStore } from "@/stores/useAppStore";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/experiments", icon: FlaskConical, label: "Experiments" },
  { to: "/classify", icon: ScanSearch, label: "Classify" },
  { to: "/embeddings", icon: Box, label: "Embeddings" },
];

export function Sidebar() {
  const { sidebarOpen, setSidebarOpen } = useAppStore();

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen border-r border-border bg-card transition-all duration-300",
        sidebarOpen ? "w-56" : "w-16"
      )}
    >
      {/* Logo */}
      <div className="flex h-14 items-center border-b border-border px-4">
        <span
          className={cn(
            "font-mono font-bold text-primary transition-all duration-300",
            sidebarOpen ? "text-lg" : "text-xs"
          )}
        >
          {sidebarOpen ? "OrchID" : "O"}
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-1 p-2">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors duration-150",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )
            }
          >
            <Icon className="h-5 w-5 shrink-0" />
            {sidebarOpen && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="absolute bottom-4 left-0 right-0 flex items-center justify-center text-muted-foreground hover:text-foreground"
        aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
      >
        {sidebarOpen ? (
          <ChevronLeft className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
      </button>
    </aside>
  );
}
