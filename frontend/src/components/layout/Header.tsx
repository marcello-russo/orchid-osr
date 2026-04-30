// frontend/src/components/layout/Header.tsx

import { Menu } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { LanguageToggle } from "./LanguageToggle";
import { useAppStore } from "@/stores/useAppStore";

export function Header() {
  const { sidebarOpen, toggleSidebar } = useAppStore();

  return (
    <header className="fixed top-0 right-0 left-0 z-30 flex h-14 items-center justify-between border-b border-border bg-card/80 backdrop-blur-md px-4">
      <button
        onClick={toggleSidebar}
        className="cursor-pointer flex items-center justify-center rounded-lg p-2 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
        aria-label={sidebarOpen ? "Close menu" : "Open menu"}
        aria-expanded={sidebarOpen}
        aria-controls="sidebar"
      >
        <Menu className="size-5" />
      </button>

      <div className="flex items-center gap-2">
        <div className="flex items-center justify-center size-7 rounded-md bg-primary/10">
          <span className="text-sm">🌸</span>
        </div>
        <span className="font-black text-base md:text-lg tracking-tighter">
          <span className="text-primary">Orch</span><span className="text-foreground">ID</span><span className="text-muted-foreground font-bold text-xs ml-0.5">NCD</span>
        </span>
      </div>

      <div className="flex items-center gap-1">
        <LanguageToggle />
        <ThemeToggle />
      </div>
    </header>
  );
}
