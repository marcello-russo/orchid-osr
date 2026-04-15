// frontend/src/components/layout/Header.tsx

import { ThemeToggle } from "./ThemeToggle";
import { useAppStore } from "@/stores/useAppStore";
import { cn } from "@/lib/utils";

export function Header() {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen);

  return (
    <header
      className={cn(
        "fixed top-0 right-0 z-30 flex h-14 items-center justify-end border-b border-border bg-card px-6 transition-all duration-300",
        sidebarOpen ? "left-56" : "left-16"
      )}
    >
      <ThemeToggle />
    </header>
  );
}
