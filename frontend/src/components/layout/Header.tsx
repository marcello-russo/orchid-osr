import { Menu } from "lucide-react";
import { useAppStore } from "@/stores/useAppStore";
import { useTranslation } from "react-i18next";

export function Header() {
  const { t } = useTranslation();
  const { sidebarOpen, toggleSidebar } = useAppStore();

  return (
    <header className="fixed top-0 right-0 left-0 z-30 flex h-14 items-center justify-between border-b border-border bg-card/80 backdrop-blur-md px-4">
      <button
        onClick={toggleSidebar}
        className="cursor-pointer flex items-center justify-center rounded-lg p-2 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
        aria-label={sidebarOpen ? t("common.close_menu") : t("common.open_menu")}
        aria-expanded={sidebarOpen}
        aria-controls="sidebar"
      >
        <Menu className="size-5" />
      </button>

      <div className="flex items-center gap-2">
        <div className="flex items-center justify-center size-7">
          <img src="/favicon.svg" alt="" className="size-6" />
        </div>
        <span className="font-black text-base md:text-lg tracking-tighter">
          <span className="text-primary">Orch</span><span className="text-foreground">ID</span><span className="text-foreground font-bold text-xs ml-0.5"> OSR</span>
        </span>
      </div>

      <div className="w-10" />
    </header>
  );
}
