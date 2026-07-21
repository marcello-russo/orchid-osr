// frontend/src/components/layout/Sidebar.tsx

import { NavLink, Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  LayoutDashboard,
  FlaskConical,
  ScanSearch,
  Box,
  Database,
  BookOpen,
  KeyRound,
  X,
  LogIn,
} from "lucide-react";
import { useAppStore } from "@/stores/useAppStore";
import { cn } from "@/lib/utils";
import { LanguageToggle } from "./LanguageToggle";
import { ThemeToggle } from "./ThemeToggle";

const navItems = [
  { to: "/", icon: LayoutDashboard, key: "common.home" },
  { to: "/experiments", icon: FlaskConical, key: "common.experiments" },
  { to: "/classify", icon: ScanSearch, key: "common.classify" },
  { to: "/dataset", icon: Database, key: "common.dataset" },
  { to: "/embeddings", icon: Box, key: "common.embeddings" },
  { to: "/research", icon: BookOpen, key: "common.about" },
];

export function Sidebar() {
  const { user, setSidebarOpen, token, logout } = useAppStore();
  const { t } = useTranslation();

  return (
    <aside className="flex h-full flex-col bg-card">
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center size-7">
            <img src="/favicon.svg" alt="" className="size-6" />
          </div>
          <span className="font-black text-lg tracking-tighter">
            <span className="text-primary">Orch</span><span className="text-foreground">ID</span><span className="text-muted-foreground font-bold text-xs ml-0.5">NCD</span>
          </span>
        </div>
        <button
          onClick={() => setSidebarOpen(false)}
          className="flex items-center justify-center rounded-lg p-2 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
          aria-label={t("common.close_menu")}
        >
          <X className="size-4" />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-1 p-2">
        {navItems.map(({ to, icon: Icon, key }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            onClick={() => setSidebarOpen(false)}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors duration-150 cursor-pointer",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )
            }
          >
            <Icon className="size-5 shrink-0" />
            <span>{t(key)}</span>
          </NavLink>
        ))}
        {user?.role === "admin" && (
          <NavLink
            to="/invites"
            onClick={() => setSidebarOpen(false)}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors duration-150 cursor-pointer",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )
            }
          >
            <KeyRound className="size-5 shrink-0" />
            <span>{t("invites.sidebar")}</span>
          </NavLink>
        )}
      </nav>

      {/* Footer */}
      <div className="mt-auto border-t border-border p-3 space-y-3">
        {token ? (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20 text-sm font-semibold text-primary">
                {user!.username.charAt(0).toUpperCase()}
              </div>
              <span className="text-sm font-medium text-foreground">{user!.username}</span>
            </div>
            <button
              onClick={() => { logout(); setSidebarOpen(false); }}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              {t("header.logout")}
            </button>
          </div>
        ) : (
          <Link
            to="/auth"
            onClick={() => setSidebarOpen(false)}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <LogIn className="h-4 w-4" />
            <span>{t("header.login")}</span>
          </Link>
        )}

        <div className="flex items-center gap-2">
          <LanguageToggle />
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}
