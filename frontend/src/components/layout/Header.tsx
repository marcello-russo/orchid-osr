// frontend/src/components/layout/Header.tsx

import { Menu } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { useAppStore } from "@/stores/useAppStore";
import { cn } from "@/lib/utils";

interface HeaderProps {
  isDesktop: boolean;
}

export function Header({ isDesktop }: HeaderProps) {
  const { sidebarOpen, toggleSidebar } = useAppStore();

  if (!isDesktop) {
    return (
      <header className="fixed top-0 right-0 left-0 z-30 flex h-14 items-center justify-between border-b border-border bg-card px-4">
        <button
          onClick={toggleSidebar}
          className="flex items-center justify-center rounded-md p-2 text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Toggle navigation menu"
        >
          <Menu className="size-5" />
        </button>
        <span className="font-mono font-bold text-primary text-lg">
          🌸 OrchID
        </span>
        <ThemeToggle />
      </header>
    );
  }

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
