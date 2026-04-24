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

interface SidebarProps {
  isDesktop: boolean;
}

export function Sidebar({ isDesktop }: SidebarProps) {
  const { sidebarOpen, setSidebarOpen, toggleSidebar } = useAppStore();

  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r border-border bg-card",
        isDesktop && "fixed left-0 top-0 z-40 h-screen",
        isDesktop && (sidebarOpen ? "w-56" : "w-16"),
        isDesktop && "transition-all duration-300"
      )}
    >
      {/* Logo — desktop only (mobile shows it in Header) */}
      {isDesktop && (
        <div className="flex h-14 items-center border-b border-border px-4">
          <span
            className={cn(
              "font-mono font-bold text-primary transition-all duration-300",
              sidebarOpen ? "text-lg" : "text-xs"
            )}
          >
            {sidebarOpen ? "🌸 OrchID" : "🌸"}
          </span>
        </div>
      )}

      {/* Mobile header */}
      {!isDesktop && (
        <div className="flex h-14 items-center border-b border-border px-4">
          <span className="font-mono font-bold text-primary text-lg">
            Navigation
          </span>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex flex-col gap-1 p-2">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            onClick={() => {
              if (!isDesktop) setSidebarOpen(false);
            }}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors duration-150",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )
            }
          >
            <Icon className="size-5 shrink-0" />
            {(isDesktop ? sidebarOpen : true) && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle — desktop only */}
      {isDesktop && (
        <button
          onClick={toggleSidebar}
          className="mt-auto flex items-center justify-center p-3 text-muted-foreground hover:text-foreground transition-colors"
          aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
        >
          {sidebarOpen ? (
            <ChevronLeft className="size-4" />
          ) : (
            <ChevronRight className="size-4" />
          )}
        </button>
      )}
    </aside>
  );
}
